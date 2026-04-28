from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from jepa_rl.browser.action_spaces import KeyboardAction
from jepa_rl.envs.browser_game_env import BrowserGameEnv, Observation, StepResult
from jepa_rl.envs.wrappers import apply_crop, apply_normalize
from jepa_rl.utils.config import ProjectConfig
from jepa_rl.utils.video import EpisodeRecorder

logger = logging.getLogger(__name__)


class BrowserEnvError(RuntimeError):
    """Raised when the Playwright browser environment cannot run."""


class BrowserClosedError(BrowserEnvError):
    """Raised when the user closes the visible browser window."""


def resolve_game_url(url: str, *, cwd: Path | None = None) -> str:
    """Resolve file URLs with repo-relative paths into absolute file URIs."""

    root = cwd or Path.cwd()
    parsed = urlparse(url)
    if parsed.scheme == "file":
        raw_path = parsed.netloc + parsed.path
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / raw_path
        return path.resolve().as_uri()

    maybe_path = Path(url)
    if parsed.scheme == "" and maybe_path.exists():
        return maybe_path.resolve().as_uri()
    return url


class PlaywrightBrowserGameEnv(BrowserGameEnv):
    """Playwright-backed browser-game environment for the first local smoke runs."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        headless: bool | None = None,
        run_dir: Path | None = None,
        record_video: bool = False,
    ):
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise BrowserEnvError(
                "Playwright is not installed. Run `uv sync` or `uv run --extra browser ...`."
            ) from exc

        self.config = config
        self.action_space = config.actions.build_discrete_keyboard_space()
        self._sync_playwright = sync_playwright
        self._playwright_error = PlaywrightError
        self._playwright_timeout_error = PlaywrightTimeoutError
        self._headless = config.game.headless if headless is None else headless
        self._run_dir = run_dir
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=config.observation.frame_stack)
        self._last_score = 0.0
        self._steps = 0
        self._recorder: EpisodeRecorder | None = None
        self._episode_index = 0
        self._reset_failures: int = 0
        self._score_failures: int = 0
        if record_video or config.recording.enabled:
            self._recorder = EpisodeRecorder(fps=config.recording.fps)

    def __enter__(self) -> PlaywrightBrowserGameEnv:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self) -> None:
        if self._page is not None:
            return
        self._playwright = self._sync_playwright().start()
        launch_kwargs = {
            "headless": self._headless,
            "args": ["--disable-extensions"],
        }
        try:
            self._browser = self._playwright.chromium.launch(**launch_kwargs)
        except self._playwright_error as exc:
            self._playwright.stop()
            raise BrowserEnvError(
                "Chromium is not installed for Playwright. Run "
                "`uv run playwright install chromium` once, then retry."
            ) from exc

        self._context = self._browser.new_context(
            viewport={
                "width": max(640, self.config.observation.width),
                "height": max(520, self.config.observation.height + 80),
            },
            device_scale_factor=1,
            record_video_dir=None,
        )
        self._page = self._context.new_page()

        if self._run_dir is not None and self.config.reward.privileged:
            self._write_run_metadata(self._run_dir)

        self.reset()

    def _write_run_metadata(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "privileged_score_reader": True,
            "game": self.config.game.name,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        metadata_path = run_dir / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def close(self) -> None:
        # Save final episode recording before closing.
        if self._recorder is not None and self._recorder.frame_count > 0:
            save_dir = self._recording_save_dir()
            self._recorder.save(save_dir)
            self._recorder.reset()

        for resource in (self._context, self._browser):
            if resource is not None:
                with suppress(self._playwright_error):
                    resource.close()
        if self._playwright is not None:
            with suppress(self._playwright_error):
                self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    def reset(self) -> Observation:
        page = self._require_page()

        # Auto-save previous episode recording before resetting.
        if self._recorder is not None and self._recorder.frame_count > 0:
            save_dir = self._recording_save_dir()
            self._recorder.save(save_dir)
            self._episode_index += 1
            self._recorder.reset()

        game_url = resolve_game_url(self.config.game.url)
        try:
            page.goto(game_url, wait_until="load")
            if self.config.reward.score_selector:
                page.wait_for_selector(
                    self.config.reward.score_selector,
                    timeout=self.config.game.reset_timeout_sec * 1000,
                )
        except Exception as exc:
            self._reset_failures += 1
            logger.warning(
                "Reset failed for game URL %s (timeout=%.1fs): %s",
                game_url,
                self.config.game.reset_timeout_sec,
                exc,
            )
            raise
        self._focus_game()
        self._steps = 0
        self._last_score = self.read_score()
        frame = self._capture_frame()
        self._frames.clear()
        for _ in range(self.config.observation.frame_stack):
            self._frames.append(frame)
        return self._observation_from_stack()

    def step(self, action: int) -> StepResult:
        selected = self.action_space.get(action)
        self._apply_action(selected)
        self._steps += 1
        self._frames.append(self._capture_frame())

        # Record full-resolution video frame if recording is active.
        if self._recorder is not None:
            self._recorder.add_frame(self.render_video_frame())

        score = self.read_score()
        reward = score - self._last_score
        self._last_score = score
        if self.config.reward.survival_bonus:
            reward += self.config.reward.survival_bonus
        done = self.is_done()
        if done and self.config.reward.death_penalty:
            reward -= abs(self.config.reward.death_penalty)
        if self.config.reward.clip_rewards:
            reward = max(-1.0, min(1.0, reward))

        return StepResult(
            observation=self._observation_from_stack(),
            reward=float(reward),
            done=done,
            score=float(score),
            info={"action_name": selected.name, "steps": self._steps},
        )

    def observe(self) -> Observation:
        if not self._frames:
            self._frames.append(self._capture_frame())
        return self._observation_from_stack()

    def read_score(self) -> float:
        if self.config.reward.score_reader != "dom":
            raise BrowserEnvError("Only DOM score reading is implemented in the first browser env.")
        selector = self.config.reward.score_selector
        if selector is None:
            raise BrowserEnvError("DOM score reader requires reward.score_selector.")
        try:
            text = self._require_page().locator(selector).inner_text(timeout=1000)
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if match is None:
                raise BrowserEnvError(f"Could not parse score from {selector!r}: {text!r}")
            return float(match.group(0))
        except Exception as exc:
            self._score_failures += 1
            logger.warning("Score reading failed at step %d: %s", self._steps, exc)
            if self._run_dir is not None:
                self._save_score_failure_screenshot()
            raise

    @property
    def reset_failures(self) -> int:
        return self._reset_failures

    @property
    def score_failures(self) -> int:
        return self._score_failures

    def is_done(self) -> bool:
        if self._steps >= self.config.game.max_steps_per_episode:
            return True
        selector = self.config.game.done_selector
        if not selector:
            return False
        return self._require_page().locator(selector).count() > 0

    def render_video_frame(self) -> np.ndarray:
        image = Image.open(BytesIO(self._require_page().screenshot())).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def wait(self, seconds: float) -> None:
        if seconds < 0:
            raise BrowserEnvError("wait seconds cannot be negative")
        page = self._require_page()
        if page.is_closed():
            raise BrowserClosedError("browser window was closed")
        try:
            page.wait_for_timeout(int(seconds * 1000))
        except self._playwright_error as exc:
            if "Target page, context or browser has been closed" in str(exc):
                raise BrowserClosedError("browser window was closed") from exc
            raise

    def save_recording(self, path: Path) -> Path | None:
        if self._recorder is None or self._recorder.frame_count == 0:
            return None
        return self._recorder.save(path)

    def _save_score_failure_screenshot(self) -> None:
        if self._run_dir is None:
            return
        try:
            failure_dir = self._run_dir / "score_failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            filename = f"step_{self._steps:06d}_{timestamp}.png"
            path = failure_dir / filename
            self._require_page().screenshot(path=str(path))
            logger.warning("Score-failure screenshot saved to %s", path)
        except Exception as screenshot_exc:
            logger.warning("Failed to save score-failure screenshot: %s", screenshot_exc)

    def _recording_save_dir(self) -> Path:
        base = self.config.recording.dir
        root = Path(base) if base else Path(self.config.experiment.output_dir) / "recordings"
        return root / f"episode_{self._episode_index:06d}"

    def _require_page(self) -> Any:
        if self._page is None:
            raise BrowserEnvError("Browser environment has not been started.")
        return self._page

    def _focus_game(self) -> None:
        page = self._require_page()
        try:
            page.locator("canvas").first.click(timeout=1000)
        except self._playwright_timeout_error:
            page.mouse.click(10, 10)

    def _apply_action(self, action: KeyboardAction) -> None:
        page = self._require_page()
        wait_ms = int(1000 * self.config.game.action_repeat / self.config.game.fps)
        for key in action.keys:
            page.keyboard.down(key)
        page.wait_for_timeout(wait_ms)
        for key in reversed(action.keys):
            page.keyboard.up(key)
        if action.is_noop:
            page.wait_for_timeout(wait_ms)
        time.sleep(0)

    def _capture_frame(self) -> np.ndarray:
        page = self._require_page()
        png = page.screenshot(full_page=False)
        image = Image.open(BytesIO(png))
        mode = "L" if self.config.observation.grayscale else "RGB"
        image = image.convert(mode)

        # Crop before resize: convert to (C, H, W) array first.
        array = np.asarray(image, dtype=np.uint8)
        array = array[None, :, :] if self.config.observation.grayscale else array.transpose(2, 0, 1)

        obs_cfg = self.config.observation
        array = apply_crop(
            array,
            top=obs_cfg.crop_top,
            bottom=obs_cfg.crop_bottom,
            left=obs_cfg.crop_left,
            right=obs_cfg.crop_right,
        )

        # Resize: convert back to PIL.
        if array.shape[0] == 1:
            image = Image.fromarray(array[0], mode="L")
        else:
            image = Image.fromarray(array.transpose(1, 2, 0), mode="RGB")
        image = image.resize(
            (self.config.observation.width, self.config.observation.height),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.uint8)
        array = array[None, :, :] if self.config.observation.grayscale else array.transpose(2, 0, 1)

        if self.config.observation.normalize:
            array = apply_normalize(array)

        return array

    def _observation_from_stack(self) -> Observation:
        data = np.concatenate(tuple(self._frames), axis=0)
        return Observation(
            data=data,
            width=self.config.observation.width,
            height=self.config.observation.height,
            channels=data.shape[0],
            metadata={"score": self._last_score, "steps": self._steps},
        )
