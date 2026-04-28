from __future__ import annotations

from jepa_rl.cli import main


def test_cli_help_exits_zero(capsys) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "JEPA-RL browser-game research CLI" in captured.out
    assert "open-game" in captured.out
    assert "ml-smoke" in captured.out
    assert "dashboard" in captured.out
    assert "ui" in captured.out


def test_cli_validate_config_exits_zero(capsys) -> None:
    exit_code = main(["validate-config", "--config", "configs/games/breakout.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "config valid" in captured.out
    assert "game=breakout" in captured.out


def test_cli_validate_config_reports_errors(capsys) -> None:
    exit_code = main(["validate-config", "--config", "tests/fixtures/configs/invalid_gamma.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "agent.gamma" in captured.err


def test_cli_open_game_rejects_negative_seconds(capsys) -> None:
    exit_code = main(
        ["open-game", "--config", "configs/games/breakout.yaml", "--seconds", "-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--seconds must be nonnegative" in captured.err


def test_cli_ml_smoke_reduces_loss(capsys) -> None:
    exit_code = main(["ml-smoke", "--steps", "400", "--lr", "0.03", "--seed", "3"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ml smoke complete" in captured.out
    assert "improvement=" in captured.out
