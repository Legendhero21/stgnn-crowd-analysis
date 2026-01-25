from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Union

import numpy as np

ArrayLike = Union[np.ndarray]


@dataclass(frozen=True)
class FeatureIndex:
    x: int = 0
    y: int = 1
    dx: int = 2
    dy: int = 3
    density: int = 4


class CrowdMetrics:
    """
    Computes interpretable crowd-level metrics from graph node features.

    Expected node feature format by default:
        x = [x, y, dx, dy, density]

    Production-oriented notes:
    - Pure, stateless API.
    - Defensive checks on shape, dtype, and NaN/Inf.
    - Configurable feature indices via `feature_index`.
    """

    DEFAULT_FEATURE_INDEX = FeatureIndex()

    @staticmethod
    def compute(
        graph: Mapping[str, ArrayLike],
        feature_index: FeatureIndex | None = None,
    ) -> Dict[str, float]:
        """
        Compute crowd metrics from a graph.

        Args:
            graph: Mapping with key "x" -> np.ndarray of shape [N, F].
            feature_index: Optional FeatureIndex to configure column positions.

        Returns:
            Dict with scalar metrics (float).
        """
        if graph is None or "x" not in graph:
            return CrowdMetrics._empty()

        x = np.asarray(graph["x"])
        if x.ndim != 2:
            return CrowdMetrics._empty()

        N, F = x.shape
        if N == 0:
            return CrowdMetrics._empty()

        fi = feature_index or CrowdMetrics.DEFAULT_FEATURE_INDEX
        max_idx = max(fi.dx, fi.dy, fi.density)
        if F <= max_idx:
            return CrowdMetrics._empty()

        # Ensure float type for safe math
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32)

        # Replace NaN/Inf with safe values
        x = np.where(np.isfinite(x), x, 0.0)

        dx = x[:, fi.dx]
        dy = x[:, fi.dy]
        density = x[:, fi.density]

        speed = np.sqrt(dx ** 2 + dy ** 2)

        # Handle degenerate all-zero case explicitly
        if speed.size == 0 or density.size == 0:
            return CrowdMetrics._empty()

        metrics = {
            "mean_speed": float(np.mean(speed)),
            "speed_std": float(np.std(speed)),
            "mean_density": float(np.mean(density)),
            "max_density": float(np.max(density)),
            "motion_entropy": CrowdMetrics._motion_entropy(dx, dy),
        }

        return metrics

    @staticmethod
    def _motion_entropy(
        dx: ArrayLike,
        dy: ArrayLike,
        bins: int = 8,
    ) -> float:
        """
        Measures disorder in motion directions.
        Higher = more chaotic movement.

        Args:
            dx: Array of x-velocities, shape [N].
            dy: Array of y-velocities, shape [N].
            bins: Number of angular bins in [-pi, pi].

        Returns:
            Scalar entropy value.
        """
        dx = np.asarray(dx)
        dy = np.asarray(dy)

        if dx.size == 0 or dy.size == 0:
            return 0.0

        # Avoid division issues with all-zero motion
        if not np.any(np.isfinite(dx)) and not np.any(np.isfinite(dy)):
            return 0.0

        angles = np.arctan2(dy, dx)

        # If all angles are identical, histogram will be very peaked
        hist, _ = np.histogram(
            angles,
            bins=bins,
            range=(-np.pi, np.pi),
            density=True,
        )

        # Guard against empty or all-zero histograms
        if hist.size == 0 or not np.any(hist):
            return 0.0

        hist = hist + 1e-6  # avoid log(0)
        entropy = -np.sum(hist * np.log(hist))
        return float(entropy)

    @staticmethod
    def _empty() -> Dict[str, float]:
        return {
            "mean_speed": 0.0,
            "speed_std": 0.0,
            "mean_density": 0.0,
            "max_density": 0.0,
            "motion_entropy": 0.0,
        }
