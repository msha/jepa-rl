from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from jepa_rl.browser.playwright_env import resolve_game_url
from jepa_rl.utils.config import load_config


def _png_bytes() -> bytes:
    image_bytes = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_bytes, format="PNG")
    return image_bytes.getvalue()


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
    score_locator = MagicMock()
    canvas_locator = MagicMock()

    sync_playwright.return_value.start.return_value = playwright
    playwright.chromium.launch.return_value = browser
    browser.new_context.return_value = context
    context.new_page.return_value = page
    score_locator.inner_text.side_effect = score_texts
    score_locator.count.return_value = 0
    canvas_locator.count.return_value = 0
    canvas_locator.first.screenshot.return_value = _png_bytes()

    def _locator(selector: str) -> MagicMock:
        if selector == "#score":
            return score_locator
        if selector == "canvas":
            return canvas_locator
        other_locator = MagicMock()
        other_locator.count.return_value = 0
        other_locator.first.screenshot.return_value = _png_bytes()
        other_locator.inner_text.return_value = ""
        return other_locator

    page.locator.side_effect = _locator

    page.screenshot.return_value = _png_bytes()
    return page, canvas_locator


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


@patch("playwright.sync_api.sync_playwright")
def test_dom_assisted_observation_adds_declared_dom_metadata(mock_sync_playwright) -> None:
    page, _ = _configure_mock_playwright(mock_sync_playwright, ["Score 0", "Score 5"])
    score_locator = MagicMock()
    score_locator.inner_text.side_effect = ["Score 0", "Score 5", "Score 5"]
    score_locator.count.return_value = 0
    lives_locator = MagicMock()
    lives_locator.inner_text.return_value = "Lives 3"

    def _locator(selector: str) -> MagicMock:
        if selector == "#score":
            return score_locator
        if selector == "#lives":
            return lives_locator
        canvas_locator = MagicMock()
        canvas_locator.first.click.return_value = None
        canvas_locator.count.return_value = 0
        return canvas_locator

    page.locator.side_effect = _locator

    from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

    config = load_config("configs/games/breakout.yaml")
    observation = replace(
        config.observation,
        mode="dom_assisted",
        dom_selectors={"score": "#score", "lives": "#lives"},
    )
    game = replace(config.game, done_selector=None)
    config = replace(config, observation=observation, game=game)

    env = PlaywrightBrowserGameEnv(config, headless=True)
    env.start()
    obs = env.observe()

    assert obs.data.shape[1:] == (config.observation.height, config.observation.width)
    assert obs.metadata["dom"]["score"]["text"] == "Score 5"
    assert obs.metadata["dom"]["score"]["value"] == 5.0
    assert obs.metadata["dom"]["lives"]["text"] == "Lives 3"
    assert obs.metadata["dom"]["lives"]["value"] == 3.0
    assert page.screenshot.called


@patch("playwright.sync_api.sync_playwright")
def test_reset_uses_configured_button_selector(mock_sync_playwright) -> None:
    page, canvas_locator = _configure_mock_playwright(mock_sync_playwright, ["0"])
    score_locator = MagicMock()
    score_locator.inner_text.return_value = "0"
    button_locator = MagicMock()

    def _locator(selector: str) -> MagicMock:
        if selector == "#score":
            return score_locator
        if selector == "#restart":
            return button_locator
        if selector == "canvas":
            return canvas_locator
        other_locator = MagicMock()
        other_locator.count.return_value = 0
        return other_locator

    page.locator.side_effect = _locator

    from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

    config = load_config("configs/games/breakout.yaml")
    game = replace(
        config.game,
        done_selector=None,
        reset_key=None,
        reset_button_selector="#restart",
    )
    config = replace(config, game=game)

    env = PlaywrightBrowserGameEnv(config, headless=True)
    env.start()

    button_locator.click.assert_called_once()
    page.keyboard.press.assert_not_called()


@patch("playwright.sync_api.sync_playwright")
def test_reset_uses_privileged_javascript_callback(mock_sync_playwright) -> None:
    page, _ = _configure_mock_playwright(mock_sync_playwright, ["0"])

    from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

    config = load_config("configs/games/breakout.yaml")
    game = replace(
        config.game,
        done_selector=None,
        reset_key=None,
        reset_javascript="window.resetGame()",
    )
    reward = replace(config.reward, privileged=True)
    config = replace(config, game=game, reward=reward)

    env = PlaywrightBrowserGameEnv(config, headless=True)
    env.start()

    page.evaluate.assert_called_once_with("window.resetGame()")
    page.keyboard.press.assert_not_called()


@patch("playwright.sync_api.sync_playwright")
def test_reset_restarts_browser_context_after_navigation_failure(mock_sync_playwright) -> None:
    playwright = MagicMock()
    browser = MagicMock()
    first_context = MagicMock()
    second_context = MagicMock()
    first_page = MagicMock()
    second_page = MagicMock()

    mock_sync_playwright.return_value.start.return_value = playwright
    playwright.chromium.launch.return_value = browser
    browser.new_context.side_effect = [first_context, second_context]
    first_context.new_page.return_value = first_page
    second_context.new_page.return_value = second_page

    first_page.goto.side_effect = RuntimeError("navigation failed")
    second_page.goto.return_value = None
    second_page.screenshot.return_value = _png_bytes()

    score_locator = MagicMock()
    score_locator.inner_text.return_value = "0"
    canvas_locator = MagicMock()
    canvas_locator.first.click.return_value = None

    def _locator(selector: str) -> MagicMock:
        if selector == "#score":
            return score_locator
        if selector == "canvas":
            return canvas_locator
        other_locator = MagicMock()
        other_locator.count.return_value = 0
        return other_locator

    second_page.locator.side_effect = _locator

    from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

    config = load_config("configs/games/breakout.yaml")
    game = replace(config.game, done_selector=None)
    config = replace(config, game=game)

    env = PlaywrightBrowserGameEnv(config, headless=True)
    env.start()

    assert env.reset_failures == 1
    first_context.close.assert_called_once()
    assert browser.new_context.call_count == 2
    second_page.goto.assert_called_once()
