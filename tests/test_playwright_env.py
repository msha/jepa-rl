from __future__ import annotations

from jepa_rl.browser.playwright_env import resolve_game_url


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
