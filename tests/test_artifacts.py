from __future__ import annotations

from jepa_rl.utils.artifacts import RUN_SUBDIRS, create_run_dir


def test_create_run_dir_creates_standard_layout(tmp_path) -> None:
    run_dir = create_run_dir(tmp_path, "experiment_a")

    assert run_dir == tmp_path / "experiment_a"
    assert run_dir.is_dir()
    for subdir in RUN_SUBDIRS:
        assert (run_dir / subdir).is_dir()
