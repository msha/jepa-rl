from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from jepa_rl.browser.playwright_env import resolve_game_url
from jepa_rl.utils.config import load_config


def test_resolve_relative_file_url_to_absolute_uri(tmp_path) -> None:
    game = tmp_path / "game.html"
    game.write_text("<html></html>", encoding="utf-8")

    resolved = resolve_game_url("file://game.html", cwd=tmp_path)

    assert resolved == game.resolve().as_uri()


def test_resolve_plain_existing_path_to_file_uri(tmp_path) -> None:
    game = tmp_path / "game.html"
    game.write_text("<html></html>", encoding="utf-8")

    resolved = resolve_game_url(str(game))

    assert resolved == game.resolve().as_uri()


def _configure_mock_playwright(
    sync_playwright: MagicMock, score_texts: list[str]
) -> tuple[MagicMock, MagicMock]:
    playwright = MagicMock()
    browser = MagicMock()
    context = MagicMock()
    page = MagicMock()
    locator = MagicMock()

    sync_playwright.return_value.start.return_value = playwright
    playwright.chromium.launch.return_value = browser
    browser.new_context.return_value = context
    context.new_page.return_value = page
    page.locator.return_value = locator
    locator.inner_text.side_effect = score_texts
    locator.count.return_value = 0

    image_bytes = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_bytes, format="PNG")
    page.screenshot.return_value = image_bytes.getvalue()
    locator.first.screenshot.return_value = image_bytes.getvalue()
    return page, locator


@patch("playwright.sync_api.sync_playwright")
def test_zero_score_penalty_applies_after_patience(mock_sync_playwright) -> None:
    _configure_mock_playwright(mock_sync_playwright, ["0", "0", "0", "0"])

    from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

    config = load_config("configs/games/breakout.yaml")
    reward = replace(config.reward, zero_score_patience_steps=1, zero_score_penalty=0.05)
    game = replace(config.game, done_selector=None)
    config = replace(config, reward=reward, game=game)

    env = PlaywrightBrowserGameEnv(config, headless=True)
    env.start()
    first = env.step(0)
    second = env.step(0)

    assert first.reward == 0.0
    assert first.info["zero_score_steps"] == 1
    assert first.info["zero_score_penalty"] == 0.0
    assert second.reward == -0.05
    assert second.info["zero_score_steps"] == 2
    assert second.info["zero_score_penalty"] == 0.05


@patch("playwright.sync_api.sync_playwright")
def test_zero_score_counter_resets_after_scoring(mock_sync_playwright) -> None:
    _configure_mock_playwright(mock_sync_playwright, ["0", "0", "10", "10"])

    from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

    config = load_config("configs/games/breakout.yaml")
    reward = replace(config.reward, zero_score_patience_steps=1, zero_score_penalty=0.05)
    game = replace(config.game, done_selector=None)
    config = replace(config, reward=reward, game=game)

    env = PlaywrightBrowserGameEnv(config, headless=True)
    env.start()
    env.step(0)
    scored = env.step(0)
    steady = env.step(0)

    assert scored.reward == 10.0
    assert scored.info["zero_score_steps"] == 0
    assert scored.info["zero_score_penalty"] == 0.0
    assert steady.reward == 0.0
    assert steady.info["zero_score_steps"] == 0


@patch("playwright.sync_api.sync_playwright")
def test_canvas_observation_mode_captures_canvas_element(mock_sync_playwright) -> None:
    page, locator = _configure_mock_playwright(mock_sync_playwright, ["0"])

    from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

    config = load_config("configs/games/breakout.yaml")
    observation = replace(config.observation, mode="canvas")
    game = replace(config.game, done_selector=None)
    config = replace(config, observation=observation, game=game)

    env = PlaywrightBrowserGameEnv(config, headless=True)
    obs = env.start() or env.observe()

    assert obs.data.shape[1:] == (config.observation.height, config.observation.width)
    assert locator.first.screenshot.called
    assert not page.screenshot.called
