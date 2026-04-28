"""Observation preprocessing wrappers for browser-game environments."""
from __future__ import annotations

import numpy as np


def apply_crop(
    obs: np.ndarray, top: int, bottom: int, left: int, right: int
) -> np.ndarray:
    """Crop pixel borders from an observation array of shape (C, H, W).

    *top*, *bottom*, *left*, *right* are the numbers of pixels to remove from
    each edge.  If all are zero the input array is returned unchanged.
    """
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return obs
    _, h, w = obs.shape
    return obs[:, top : h - bottom, left : w - right]


def apply_normalize(obs: np.ndarray) -> np.ndarray:
    """Normalize a uint8 ``[0, 255]`` array to float32 ``[0.0, 1.0]``."""
    return obs.astype(np.float32) / 255.0
