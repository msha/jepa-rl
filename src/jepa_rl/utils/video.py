"""Episode video recording for browser-game environments.

Records RGB frames and writes them as numbered PNG files inside a directory.
This avoids requiring OpenCV or imageio as dependencies while still providing
durable episode replays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class EpisodeRecorder:
    """Records RGB frames from a browser environment and writes them to disk.

    Frames are stored in-memory until ``save()`` is called, at which point they
    are written as numbered PNG files (``frame_000000.png``, ``frame_000001.png``,
    ...) inside a directory at the specified path.  The internal buffer is cleared
    after saving.
    """

    def __init__(self, fps: int = 30) -> None:
        self._fps = fps
        self._frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> None:
        """Add an RGB frame (H, W, 3) uint8 array."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must be (H, W, 3) uint8, got shape {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8, got {frame.dtype}")
        self._frames.append(frame)

    @property
    def frame_count(self) -> int:
        """Number of frames recorded so far."""
        return len(self._frames)

    @property
    def fps(self) -> int:
        return self._fps

    def save(self, path: Path) -> Path:
        """Write all accumulated frames as PNGs inside a directory at *path*.

        Creates parent directories as needed.  If no frames have been added, an
        empty directory is created.  Returns the path to the written directory.
        Clears the internal frame buffer after saving.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        meta_path = target / "meta.yaml"
        meta_path.write_text(
            f"fps: {self._fps}\nframe_count: {len(self._frames)}\n",
            encoding="utf-8",
        )

        for idx, frame in enumerate(self._frames):
            filename = f"frame_{idx:06d}.png"
            img = Image.fromarray(frame, mode="RGB")
            img.save(target / filename, format="PNG")

        self._frames.clear()
        return target

    def reset(self) -> None:
        """Discard all accumulated frames without saving."""
        self._frames.clear()
