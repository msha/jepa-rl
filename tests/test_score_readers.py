"""Tests for the DOM score reader regex and failure-handling logic."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


def test_screenshot_failure_does_not_suppress_original_error(tmp_path):
    env = _make_env_with_mock_page(tmp_path, page_text="no digits")
    env._page.screenshot.side_effect = RuntimeError("page crashed")

    with pytest.raises(BrowserEnvError, match="Could not parse score"):
        env.read_score()
