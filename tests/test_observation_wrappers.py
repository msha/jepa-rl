"""Tests for observation wrappers and frame-stack shapes."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from jepa_rl.envs.wrappers import apply_crop, apply_normalize


class TestApplyCrop:
    def test_crop_reduces_dimensions(self) -> None:
        """Cropping (C, H, W) by (10, 10, 5, 5) reduces H by 20 and W by 10."""
        obs = np.zeros((3, 100, 80), dtype=np.uint8)
        result = apply_crop(obs, top=10, bottom=10, left=5, right=5)
        assert result.shape == (3, 80, 70)

    def test_crop_zero_returns_same(self) -> None:
        """All-zero crop returns the input array unchanged."""
        obs = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
        result = apply_crop(obs, top=0, bottom=0, left=0, right=0)
        assert result is obs

    def test_crop_preserves_dtype(self) -> None:
        obs = np.ones((1, 30, 40), dtype=np.float32)
        result = apply_crop(obs, top=5, bottom=5, left=10, right=10)
        assert result.dtype == np.float32
        assert result.shape == (1, 20, 20)

    def test_crop_single_channel(self) -> None:
        obs = np.ones((1, 50, 60), dtype=np.uint8)
        result = apply_crop(obs, top=0, bottom=10, left=0, right=10)
        assert result.shape == (1, 40, 50)


class TestApplyNormalize:
    def test_normalize_range(self) -> None:
        """Normalized array has values in [0, 1] with float32 dtype."""
        obs = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = apply_normalize(obs)
        assert result.dtype == np.float32
        assert result.shape == obs.shape
        assert np.isclose(result[0, 0, 0], 0.0)
        assert np.isclose(result[0, 0, 1], 128 / 255.0)
        assert np.isclose(result[0, 0, 2], 1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_idempotent(self) -> None:
        """Normalizing an already-normalized array scales values by ~1/255."""
        obs = np.array([[[0, 64, 128, 192, 255]]], dtype=np.uint8)
        first = apply_normalize(obs)
        second = apply_normalize(first)
        expected = first / 255.0
        np.testing.assert_allclose(second, expected, atol=1e-7)

    def test_normalize_dtype(self) -> None:
        obs = np.zeros((3, 10, 10), dtype=np.uint8)
        result = apply_normalize(obs)
        assert result.dtype == np.float32


@pytest.mark.parametrize("grayscale", [True, False])
@pytest.mark.parametrize("frame_stack", [1, 4])
def test_frame_stack_shape(grayscale: bool, frame_stack: int) -> None:
    """After frame stacking, the concatenated data has shape (C*frame_stack, H, W).

    This simulates the frame-stacking pipeline used by PlaywrightBrowserGameEnv
    without requiring a running browser.
    """
    from jepa_rl.utils.config import load_config

    config = load_config("configs/games/breakout.yaml")

    channels = 1 if grayscale else 3
    h, w = config.observation.height, config.observation.width

    frames: deque[np.ndarray] = deque(maxlen=frame_stack)
    single_frame = np.random.randint(0, 256, (channels, h, w), dtype=np.uint8)
    for _ in range(frame_stack):
        frames.append(single_frame)

    stacked = np.concatenate(tuple(frames), axis=0)
    expected_channels = channels * frame_stack
    assert stacked.shape == (expected_channels, h, w)
