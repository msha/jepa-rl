"""Tests for the DOM score reader regex and failure-handling logic."""
from __future__ import annotations

import re
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from jepa_rl.browser.playwright_env import BrowserEnvError, PlaywrightBrowserGameEnv

SCORE_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def test_score_positive_integer():
    assert float(SCORE_PATTERN.search("Score: 42").group()) == 42.0


def test_score_negative():
    assert float(SCORE_PATTERN.search("Score: -5").group()) == -5.0


def test_score_float():
    assert float(SCORE_PATTERN.search("Points: 12.5").group()) == 12.5


def test_score_no_number():
    assert SCORE_PATTERN.search("no score here") is None


def test_score_embedded_in_text():
    assert float(SCORE_PATTERN.search("You got 100 points!").group()) == 100.0


def test_score_first_number_wins():
    assert float(SCORE_PATTERN.search("Level 3 Score: 150").group()) == 3.0


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "html"


def _read_score_span(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'<span id="score">(.*?)</span>', text)
    assert m is not None, f"No <span id='score'> found in {path}"
    return m.group(1)


def test_fixture_score_positive():
    value = _read_score_span(FIXTURES_DIR / "score_positive.html")
    assert float(SCORE_PATTERN.search(value).group()) == 42.0


def test_fixture_score_negative():
    value = _read_score_span(FIXTURES_DIR / "score_negative.html")
    assert float(SCORE_PATTERN.search(value).group()) == -5.0


def test_fixture_score_no_number():
    value = _read_score_span(FIXTURES_DIR / "score_no_number.html")
    assert SCORE_PATTERN.search(value) is None


def _make_env_with_mock_page(
    tmp_path: Path, *, page_text: str = "N/A"
) -> PlaywrightBrowserGameEnv:
    from jepa_rl.utils.config import load_config

    config = load_config("configs/games/breakout.yaml")

    env = object.__new__(PlaywrightBrowserGameEnv)
    env.config = config
    env.action_space = config.actions.build_discrete_keyboard_space()
    env._sync_playwright = None
    env._playwright_error = RuntimeError
    env._playwright_timeout_error = TimeoutError
    env._headless = True
    env._run_dir = tmp_path
    env._playwright = None
    env._browser = None
    env._context = None
    env._frames = []
    env._last_score = 0.0
    env._steps = 7

    mock_locator = MagicMock()
    mock_locator.inner_text.return_value = page_text

    mock_page = MagicMock()
    mock_page.locator.return_value = mock_locator

    _fake_png = b"\x89PNG\r\n\x1a\n"

    def _screenshot_side_effect(**kwargs):
        path = kwargs.get("path")
        if path is not None:
            Path(path).write_bytes(_fake_png)
        return _fake_png

    mock_page.screenshot.side_effect = _screenshot_side_effect

    env._page = mock_page
    env._recorder = None
    env._reset_failures = 0
    env._score_failures = 0

    return env


def test_screenshot_saved_on_score_failure(tmp_path):
    env = _make_env_with_mock_page(tmp_path, page_text="no digits in here")

    with pytest.raises(BrowserEnvError, match="Could not parse score"):
        env.read_score()

    failure_dir = tmp_path / "score_failures"
    assert failure_dir.is_dir(), "score_failures directory should be created"

    screenshots = list(failure_dir.glob("*.png"))
    assert len(screenshots) == 1, f"Expected exactly one screenshot, found {len(screenshots)}"
    name = screenshots[0].name
    assert name.startswith(
        "step_000007_"
    ), f"Screenshot filename should start with step_000007_, got {name}"
    assert name.endswith(".png")


def test_no_screenshot_when_run_dir_is_none(tmp_path):
    env = _make_env_with_mock_page(tmp_path, page_text="no digits")
    env._run_dir = None

    with pytest.raises(BrowserEnvError, match="Could not parse score"):
        env.read_score()

    assert not (tmp_path / "score_failures").exists()


def test_read_score_success_through_mock_page(tmp_path):
    """read_score returns the correct float when the mock page has a valid score."""
    env = _make_env_with_mock_page(tmp_path, page_text="42")

    score = env.read_score()

    assert score == 42.0


def test_screenshot_failure_does_not_suppress_original_error(tmp_path):
    env = _make_env_with_mock_page(tmp_path, page_text="no digits")
    env._page.screenshot.side_effect = RuntimeError("page crashed")

    with pytest.raises(BrowserEnvError, match="Could not parse score"):
        env.read_score()


def test_ocr_score_reader_uses_configured_score_region(tmp_path, monkeypatch):
    env = _make_env_with_mock_page(tmp_path)
    reward = replace(
        env.config.reward,
        score_reader="ocr",
        score_selector=None,
        score_region=(1, 2, 3, 4),
    )
    env.config = replace(env.config, reward=reward)

    png = BytesIO()
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(png, format="PNG")
    env._page.screenshot.side_effect = None
    env._page.screenshot.return_value = png.getvalue()

    seen_sizes: list[tuple[int, int]] = []

    def _fake_ocr(image):
        seen_sizes.append(image.size)
        return "Score 123"

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(image_to_string=_fake_ocr),
    )

    assert env.read_score() == 123.0
    assert seen_sizes == [(3, 4)]


def test_canvas_ocr_score_reader_captures_canvas(tmp_path, monkeypatch):
    env = _make_env_with_mock_page(tmp_path)
    reward = replace(env.config.reward, score_reader="canvas_ocr", score_selector=None)
    env.config = replace(env.config, reward=reward)

    png = BytesIO()
    Image.new("RGB", (6, 6), color=(255, 255, 255)).save(png, format="PNG")
    canvas_locator = MagicMock()
    canvas_locator.first.screenshot.return_value = png.getvalue()
    env._page.locator.return_value = canvas_locator

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(image_to_string=lambda image: "77"),
    )

    assert env.read_score() == 77.0
    canvas_locator.first.screenshot.assert_called_once()
