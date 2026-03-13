"""
ByteTrack Multi-Object Tracker
-------------------------------
Thin wrapper around Ultralytics' built-in .track() API.

Reference:
    Zhang et al., 2022
    "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"

Provides stable person IDs across frames so that velocity, heading, and
trajectory features can be computed reliably.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ByteTracker")


@dataclass
class TrackedPerson:
    """Single tracked person in the current frame."""

    track_id: int
    cx: float          # center x (pixels)
    cy: float          # center y (pixels)
    x1: float          # bbox left
    y1: float          # bbox top
    x2: float          # bbox right
    y2: float          # bbox bottom
    conf: float        # detection confidence
    bbox_area: float = 0.0   # bbox area in normalized coords (filled later)


class ByteTrackTracker:
    """
    Wraps Ultralytics YOLO + ByteTrack tracking.

    Usage:
        tracker = ByteTrackTracker(model_path="yolo11n.pt")
        tracked = tracker.update(frame)
        # tracked: List[TrackedPerson]
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        device: str = "cuda",
        tracker_type: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO

        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.tracker_type = tracker_type

        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            logger.error("Failed to load YOLO model from %s: %s", model_path, exc)
            raise

        try:
            self.model.to(device)
        except Exception:
            logger.warning("Could not move model to %s, falling back to CPU.", device)
            self.device = "cpu"
            self.model.to("cpu")

        # Previous tracked positions for velocity computation {track_id: (cx_norm, cy_norm)}
        self._prev_positions: Dict[int, Tuple[float, float]] = {}

        logger.info("ByteTrackTracker ready  model=%s  device=%s", model_path, self.device)

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: BGR image [H, W, 3].

        Returns:
            List of TrackedPerson instances (may be empty).
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            classes=[0],          # person class only
            persist=True,         # keep tracker state across calls
            tracker=self.tracker_type,
        )

        tracked: List[TrackedPerson] = []

        if not results or results[0].boxes is None:
            return tracked

        boxes = results[0].boxes

        # .id may be None if tracking hasn't assigned IDs yet
        if boxes.id is None:
            return tracked

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for tid, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            tracked.append(
                TrackedPerson(
                    track_id=int(tid),
                    cx=cx,
                    cy=cy,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    conf=float(conf),
                )
            )

        return tracked

    # ------------------------------------------------------------------
    def get_previous_positions(self) -> Dict[int, Tuple[float, float]]:
        """Return previous-frame normalized positions for velocity computation."""
        return dict(self._prev_positions)

    def store_positions(self, positions: Dict[int, Tuple[float, float]]) -> None:
        """Store current-frame normalized positions for the next velocity computation."""
        self._prev_positions = dict(positions)
