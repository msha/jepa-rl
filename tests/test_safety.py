"""Safety and isolation tests for the browser environment (Phase 12).

These tests verify that the Playwright browser environment enforces isolation
constraints: extensions are disabled, privileged callbacks are recorded in
run metadata, headless mode is the default, and viewport dimensions match
config.

All tests mock the Playwright sync API so they run without a real browser.
"""
from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from jepa_rl.utils.config import load_config


def _make_mock_playwright_chain():
    mock_sync_pw = MagicMock()
    mock_pw_instance = MagicMock()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()

    mock_sync_pw.return_value.start.return_value = mock_pw_instance
    mock_pw_instance.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    mock_locator = MagicMock()
    mock_locator.inner_text.return_value = "0"
    mock_page.locator.return_value = mock_locator
    mock_page.locator.return_value.count.return_value = 0

    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    mock_page.screenshot.return_value = buf.getvalue()

    return mock_sync_pw, mock_pw_instance, mock_browser, mock_context, mock_page


_PW_PATCH_TARGET = "playwright.sync_api.sync_playwright"


def _load_breakout_config():
    return load_config("configs/games/breakout.yaml")


class TestExtensionDisabling:
    @patch(_PW_PATCH_TARGET)
    def test_launch_args_include_disable_extensions(self, mock_sync_pw):
        _, mock_pw, *_ = _make_mock_playwright_chain()
        mock_sync_pw.return_value.start.return_value = mock_pw

        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        config = _load_breakout_config()
        env = PlaywrightBrowserGameEnv(config, headless=True)
        env.start()

        launch_call = mock_pw.chromium.launch.call_args
        args = launch_call.kwargs.get("args", [])
        assert "--disable-extensions" in args, (
            f"Expected --disable-extensions in launch args, got {args}"
        )

    @patch(_PW_PATCH_TARGET)
    def test_launch_passes_headless_flag(self, mock_sync_pw):
        _, mock_pw, *_ = _make_mock_playwright_chain()
        mock_sync_pw.return_value.start.return_value = mock_pw

        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        config = _load_breakout_config()
        env = PlaywrightBrowserGameEnv(config, headless=True)
        env.start()

        launch_call = mock_pw.chromium.launch.call_args
        assert launch_call.kwargs.get("headless") is True

    @patch(_PW_PATCH_TARGET)
    def test_launch_args_is_explicit_list(self, mock_sync_pw):
        _, mock_pw, *_ = _make_mock_playwright_chain()
        mock_sync_pw.return_value.start.return_value = mock_pw

        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        config = _load_breakout_config()
        env = PlaywrightBrowserGameEnv(config, headless=True)
        env.start()

        launch_call = mock_pw.chromium.launch.call_args
        assert "args" in launch_call.kwargs, "Missing 'args' in launch kwargs"
        assert isinstance(launch_call.kwargs["args"], list)


class TestFixedViewport:
    @patch(_PW_PATCH_TARGET)
    def test_viewport_matches_config(self, mock_sync_pw):
        _, mock_pw, mock_browser, *_ = _make_mock_playwright_chain()
        mock_sync_pw.return_value.start.return_value = mock_pw

        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        config = _load_breakout_config()
        env = PlaywrightBrowserGameEnv(config, headless=True)
        env.start()

        context_call = mock_browser.new_context.call_args
        viewport = context_call.kwargs.get("viewport", {})
        assert viewport["width"] >= config.observation.width
        assert viewport["height"] >= config.observation.height

    @patch(_PW_PATCH_TARGET)
    def test_viewport_is_always_set(self, mock_sync_pw):
        _, mock_pw, mock_browser, *_ = _make_mock_playwright_chain()
        mock_sync_pw.return_value.start.return_value = mock_pw

        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        config = _load_breakout_config()
        env = PlaywrightBrowserGameEnv(config, headless=True)
        env.start()

        context_call = mock_browser.new_context.call_args
        viewport = context_call.kwargs.get("viewport")
        assert viewport is not None
        assert "width" in viewport
        assert "height" in viewport


class TestHeadlessDefault:
    def test_base_config_headless_true(self):
        config = load_config("configs/base.yaml")
        assert config.game.headless is True

    def test_breakout_config_headless_true(self):
        config = _load_breakout_config()
        assert config.game.headless is True


class TestNoPrivilegedWithoutFlag:
    def test_default_config_privileged_false(self):
        config = _load_breakout_config()
        assert config.reward.privileged is False

    @patch(_PW_PATCH_TARGET)
    def test_run_metadata_not_written_without_privileged(self, mock_sync_pw, tmp_path):
        _, mock_pw, *_ = _make_mock_playwright_chain()
        mock_sync_pw.return_value.start.return_value = mock_pw

        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        config = _load_breakout_config()
        run_dir = tmp_path / "run"
        env = PlaywrightBrowserGameEnv(config, headless=True, run_dir=run_dir)
        env.start()

        assert not (run_dir / "run_metadata.json").exists()

    @patch(_PW_PATCH_TARGET)
    def test_run_metadata_written_with_privileged(self, mock_sync_pw, tmp_path):
        _, mock_pw, *_ = _make_mock_playwright_chain()
        mock_sync_pw.return_value.start.return_value = mock_pw

        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        config = _load_breakout_config()
        privileged_reward = replace(config.reward, privileged=True)
        config = replace(config, reward=privileged_reward)

        run_dir = tmp_path / "run"
        env = PlaywrightBrowserGameEnv(config, headless=True, run_dir=run_dir)
        env.start()

        metadata_path = run_dir / "run_metadata.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["privileged_score_reader"] is True
        assert metadata["game"] == config.game.name
        assert "timestamp" in metadata

    @patch(_PW_PATCH_TARGET)
    def test_run_metadata_not_written_without_run_dir(self, mock_sync_pw, tmp_path):
        _, mock_pw, *_ = _make_mock_playwright_chain()
        mock_sync_pw.return_value.start.return_value = mock_pw

        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        config = _load_breakout_config()
        privileged_reward = replace(config.reward, privileged=True)
        config = replace(config, reward=privileged_reward)

        env = PlaywrightBrowserGameEnv(config, headless=True)
        env.start()

        json_files = list(tmp_path.rglob("run_metadata.json"))
        assert json_files == []


class TestConfigSafetyDefaults:
    def test_no_javascript_score_reader_by_default(self):
        config = _load_breakout_config()
        assert config.reward.score_reader == "dom"

    def test_javascript_reader_requires_privileged(self):
        from jepa_rl.utils.config import ConfigError, RewardConfig

        with pytest.raises(ConfigError, match="javascript"):
            RewardConfig.from_dict({
                "type": "score_delta",
                "score_reader": "javascript",
                "score_selector": "#score",
                "privileged": False,
            })

    def test_privileged_default_in_reward_config(self):
        from jepa_rl.utils.config import RewardConfig

        reward = RewardConfig.from_dict({
            "type": "score_delta",
            "score_reader": "dom",
            "score_selector": "#score",
        })
        assert reward.privileged is False
