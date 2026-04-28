"""Tests for the EpisodeRecorder (src/jepa_rl/utils/video.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jepa_rl.utils.video import EpisodeRecorder


def _make_frame(h: int = 16, w: int = 16) -> np.ndarray:
    """Return a deterministic (H, W, 3) uint8 RGB frame."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


class TestEpisodeRecorderAccumulatesFrames:
    def test_frame_count_after_add(self):
        rec = EpisodeRecorder(fps=30)
        for _ in range(5):
            rec.add_frame(_make_frame())
        assert rec.frame_count == 5


class TestEpisodeRecorderSaveCreatesFile:
    def test_save_creates_directory_with_pngs(self, tmp_path: Path):
        rec = EpisodeRecorder(fps=30)
        for _ in range(3):
            rec.add_frame(_make_frame())
        out = tmp_path / "ep_001"
        result = rec.save(out)
        assert result == out
        assert out.is_dir()
        pngs = sorted(out.glob("frame_*.png"))
        assert len(pngs) == 3
        assert (out / "meta.yaml").exists()


class TestEpisodeRecorderSaveClearsBuffer:
    def test_frame_count_zero_after_save(self, tmp_path: Path):
        rec = EpisodeRecorder(fps=30)
        for _ in range(4):
            rec.add_frame(_make_frame())
        rec.save(tmp_path / "ep")
        assert rec.frame_count == 0


class TestEpisodeRecorderResetDiscardsFrames:
    def test_reset_clears_without_writing(self, tmp_path: Path):
        rec = EpisodeRecorder(fps=30)
        for _ in range(6):
            rec.add_frame(_make_frame())
        rec.reset()
        assert rec.frame_count == 0
        assert list(tmp_path.iterdir()) == []


class TestEpisodeRecorderEmptyEpisode:
    def test_save_with_zero_frames_does_not_crash(self, tmp_path: Path):
        rec = EpisodeRecorder(fps=30)
        out = tmp_path / "empty_ep"
        result = rec.save(out)
        assert result == out
        assert out.is_dir()
        pngs = list(out.glob("frame_*.png"))
        assert len(pngs) == 0
        meta = out / "meta.yaml"
        assert meta.exists()
        assert "frame_count: 0" in meta.read_text()


class TestEpisodeRecorderCorrectFrameCount:
    def test_written_pngs_match_added_frames(self, tmp_path: Path):
        rec = EpisodeRecorder(fps=15)
        n = 10
        for i in range(n):
            frame = np.full((8, 8, 3), fill_value=i % 256, dtype=np.uint8)
            rec.add_frame(frame)

        out = tmp_path / "ordered"
        rec.save(out)

        pngs = sorted(out.glob("frame_*.png"))
        assert len(pngs) == n

        from PIL import Image

        for idx, png_path in enumerate(pngs):
            img = Image.open(png_path)
            arr = np.asarray(img)
            expected = idx % 256
            assert arr[0, 0, 0] == expected


class TestEpisodeRecorderRejectsBadFrames:
    def test_rejects_grayscale(self):
        rec = EpisodeRecorder(fps=30)
        with pytest.raises(ValueError, match="must be \\(H, W, 3\\)"):
            rec.add_frame(np.zeros((16, 16), dtype=np.uint8))

    def test_rejects_float_dtype(self):
        rec = EpisodeRecorder(fps=30)
        with pytest.raises(ValueError, match="dtype must be uint8"):
            rec.add_frame(np.zeros((16, 16, 3), dtype=np.float32))
