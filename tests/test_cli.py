from __future__ import annotations

from jepa_rl.cli import main


def test_cli_help_exits_zero(capsys) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "JEPA-RL browser-game research CLI" in captured.out


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
