import logging
from collections import deque
from typing import Deque, Dict

import numpy as np

logger = logging.getLogger(__name__)


class StampedeAlert:
    """
    Deterministic, rule-based stampede alert logic.
    No learning; thresholds are explicit and thesis-audit friendly.
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 15,
        density_warn: float = 0.45,
        density_crit: float = 0.65,
        entropy_warn: float = 0.60,
        entropy_crit: float = 0.85,
        anomaly_crit: float = 0.08,
    ) -> None:
        """
        All thresholds must be consistent with your pipeline scales.
        Tune once using validation data, then freeze for thesis.
        """
        if short_window <= 1 or long_window <= short_window:
            raise ValueError("Require 1 < short_window < long_window.")

        self.anomaly_hist: Deque[float] = deque(maxlen=long_window)
        self.density_hist: Deque[float] = deque(maxlen=long_window)
        self.entropy_hist: Deque[float] = deque(maxlen=long_window)

        self.short_window = short_window

        self.density_warn = float(density_warn)
        self.density_crit = float(density_crit)
        self.entropy_warn = float(entropy_warn)
        self.entropy_crit = float(entropy_crit)
        self.anomaly_crit = float(anomaly_crit)

    def update(self, anomaly_score: float, metrics: Dict[str, float]) -> str:
        """
        Inputs:
            anomaly_score: float (STGNN MSE).
            metrics: dict from crowd_metrics.CrowdMetrics.compute.

        Output:
            state: "NORMAL" | "UNSTABLE" | "STAMPEDE"
        """
        mean_density = float(metrics.get("mean_density", 0.0))
        motion_entropy = float(metrics.get("motion_entropy", 0.0))

        # Store history
        self.anomaly_hist.append(float(anomaly_score))
        self.density_hist.append(mean_density)
        self.entropy_hist.append(motion_entropy)

        # Not enough temporal context yet
        if len(self.anomaly_hist) < self.short_window:
            return "NORMAL"

        # Short-term trends (last short_window-1 diffs)
        a_arr = np.asarray(self.anomaly_hist, dtype=np.float32)
        d_arr = np.asarray(self.density_hist, dtype=np.float32)

        if a_arr.size > 1:
            a_trend = float(np.mean(np.diff(a_arr[-self.short_window :])))
        else:
            a_trend = 0.0

        if d_arr.size > 1:
            d_trend = float(np.mean(np.diff(d_arr[-self.short_window :])))
        else:
            d_trend = 0.0

        # --- STAMPEDE RULE (STRICT) ---
        if (
            mean_density > self.density_crit
            and motion_entropy > self.entropy_crit
            and anomaly_score > self.anomaly_crit
        ):
            return "STAMPEDE"

        # --- UNSTABLE RULE (PRE-ALERT) ---
        unstable_signals = 0

        if mean_density > self.density_warn:
            unstable_signals += 1
        if motion_entropy > self.entropy_warn:
            unstable_signals += 1
        if a_trend > 0:
            unstable_signals += 1
        if d_trend > 0:
            unstable_signals += 1

        if unstable_signals >= 2:
            return "UNSTABLE"

        return "NORMAL"
