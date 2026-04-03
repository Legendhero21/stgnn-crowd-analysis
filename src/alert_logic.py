import logging
from collections import deque
from typing import Deque, Dict

import numpy as np

logger = logging.getLogger(__name__)


class StampedeAlert:
    """
    Deterministic alert state machine for real-time crowd monitoring.

    Design goals:
    - Require sustained evidence before escalating.
    - Use hysteresis to avoid alert flicker.
    - Gate alerts on crowd size so sparse normal scenes stay normal.
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 15,
        min_people_unstable: int = 6,
        min_people_high_alert: int = 10,
        unstable_confirm_frames: int = 3,
        high_alert_confirm_frames: int = 4,
        unstable_clear_frames: int = 4,
        high_alert_clear_frames: int = 3,
        density_warn_enter: float = 0.55,
        density_warn_exit: float = 0.45,
        density_high_enter: float = 0.70,
        density_high_exit: float = 0.55,
        entropy_warn_enter: float = 1.15,
        entropy_warn_exit: float = 0.95,
        entropy_high_enter: float = 1.60,
        entropy_high_exit: float = 1.20,
        anomaly_warn_enter: float = 0.70,
        anomaly_warn_exit: float = 0.60,
        anomaly_high_enter: float = 0.95,
        anomaly_high_exit: float = 0.85,
        speed_warn_enter: float = 0.004,
        speed_warn_exit: float = 0.0025,
        speed_high_enter: float = 0.010,
        speed_high_exit: float = 0.007,
        anomaly_trend_enter: float = 0.03,
        density_trend_enter: float = 0.05,
    ) -> None:
        """
        All thresholds must be consistent with your pipeline scales.
        Tune once using validation data, then freeze for thesis.
        """
        if short_window <= 1 or long_window <= short_window:
            raise ValueError("Require 1 < short_window < long_window.")
        if min_people_unstable < 1 or min_people_high_alert < min_people_unstable:
            raise ValueError(
                "Require 1 <= min_people_unstable <= min_people_high_alert."
            )

        self.anomaly_hist: Deque[float] = deque(maxlen=long_window)
        self.density_hist: Deque[float] = deque(maxlen=long_window)
        self.entropy_hist: Deque[float] = deque(maxlen=long_window)
        self.people_hist: Deque[float] = deque(maxlen=long_window)

        self.short_window = short_window

        self.min_people_unstable = int(min_people_unstable)
        self.min_people_high_alert = int(min_people_high_alert)

        self.unstable_confirm_frames = int(unstable_confirm_frames)
        self.high_alert_confirm_frames = int(high_alert_confirm_frames)
        self.unstable_clear_frames = int(unstable_clear_frames)
        self.high_alert_clear_frames = int(high_alert_clear_frames)

        self.density_warn_enter = float(density_warn_enter)
        self.density_warn_exit = float(density_warn_exit)
        self.density_high_enter = float(density_high_enter)
        self.density_high_exit = float(density_high_exit)

        self.entropy_warn_enter = float(entropy_warn_enter)
        self.entropy_warn_exit = float(entropy_warn_exit)
        self.entropy_high_enter = float(entropy_high_enter)
        self.entropy_high_exit = float(entropy_high_exit)

        self.anomaly_warn_enter = float(anomaly_warn_enter)
        self.anomaly_warn_exit = float(anomaly_warn_exit)
        self.anomaly_high_enter = float(anomaly_high_enter)
        self.anomaly_high_exit = float(anomaly_high_exit)

        self.speed_warn_enter = float(speed_warn_enter)
        self.speed_warn_exit = float(speed_warn_exit)
        self.speed_high_enter = float(speed_high_enter)
        self.speed_high_exit = float(speed_high_exit)

        self.anomaly_trend_enter = float(anomaly_trend_enter)
        self.density_trend_enter = float(density_trend_enter)

        self._state = "NORMAL"
        self._unstable_candidate_frames = 0
        self._high_alert_candidate_frames = 0
        self._calm_frames = 0

    def _recent_mean(self, values: Deque[float]) -> float:
        recent = list(values)[-self.short_window :]
        return float(np.mean(recent)) if recent else 0.0

    def _recent_trend(self, values: Deque[float]) -> float:
        recent = np.asarray(list(values)[-self.short_window :], dtype=np.float32)
        if recent.size < 2:
            return 0.0
        return float(recent[-1] - recent[0])

    def _reset_candidates(self) -> None:
        self._unstable_candidate_frames = 0
        self._high_alert_candidate_frames = 0

    def _mark_calm(self) -> None:
        self._calm_frames += 1
        self._reset_candidates()

    def _mark_active(self) -> None:
        self._calm_frames = 0

    def update(self, anomaly_score: float, metrics: Dict[str, float]) -> str:
        """
        Inputs:
            anomaly_score: float (STGNN MSE).
            metrics: dict from crowd_metrics.CrowdMetrics.compute.

        Output:
            state: "NORMAL" | "UNSTABLE" | "HIGH_ALERT"
        """
        mean_density = float(metrics.get("mean_density", 0.0))
        motion_entropy = float(metrics.get("motion_entropy", 0.0))
        mean_speed = float(metrics.get("mean_speed", 0.0))
        active_nodes = int(round(float(metrics.get("active_nodes", 0.0))))

        # Store history
        self.anomaly_hist.append(float(anomaly_score))
        self.density_hist.append(mean_density)
        self.entropy_hist.append(motion_entropy)
        self.people_hist.append(float(active_nodes))

        # Not enough temporal context yet
        if len(self.anomaly_hist) < self.short_window:
            return "NORMAL"

        smoothed_density = self._recent_mean(self.density_hist)
        smoothed_entropy = self._recent_mean(self.entropy_hist)
        smoothed_anomaly = self._recent_mean(self.anomaly_hist)
        smoothed_people = int(round(self._recent_mean(self.people_hist)))
        smoothed_speed = mean_speed

        # The dashboard and operator policy use anomaly-score bands directly:
        # < 0.70 => NORMAL, 0.70-0.95 => at most UNSTABLE, >= 0.95 => HIGH_ALERT candidate.
        # Enforce those bands using the current displayed score so the UI state and
        # operator-visible anomaly value do not contradict each other.
        if anomaly_score < self.anomaly_warn_enter:
            self._state = "NORMAL"
            self._calm_frames = 0
            self._reset_candidates()
            return self._state

        anomaly_trend = self._recent_trend(self.anomaly_hist)
        density_trend = self._recent_trend(self.density_hist)

        high_density = smoothed_density >= self.density_high_enter
        high_entropy = smoothed_entropy >= self.entropy_high_enter
        high_speed = smoothed_speed >= self.speed_high_enter
        high_anomaly = smoothed_anomaly >= self.anomaly_high_enter

        warn_density = smoothed_density >= self.density_warn_enter
        warn_entropy = smoothed_entropy >= self.entropy_warn_enter
        warn_speed = smoothed_speed >= self.speed_warn_enter
        warn_anomaly = smoothed_anomaly >= self.anomaly_warn_enter
        rising_risk = (
            anomaly_trend >= self.anomaly_trend_enter
            or density_trend >= self.density_trend_enter
        )

        contextual_warn = sum(
            [
                warn_density,
                warn_entropy,
                warn_speed,
                rising_risk,
            ]
        )

        in_high_alert_band = anomaly_score >= self.anomaly_high_enter
        in_unstable_band = (
            self.anomaly_warn_enter <= anomaly_score < self.anomaly_high_enter
        )

        qualifies_high_alert = (
            in_high_alert_band
            and
            smoothed_people >= self.min_people_high_alert
            and high_anomaly
            and high_density
            and (high_entropy or high_speed)
        )
        qualifies_unstable = (
            in_unstable_band
            and
            smoothed_people >= self.min_people_unstable
            and warn_anomaly
            and contextual_warn >= 1
        )

        calm_for_high_alert = (
            not in_high_alert_band
            or smoothed_people < self.min_people_high_alert
            or (
                smoothed_density <= self.density_high_exit
                and smoothed_entropy <= self.entropy_high_exit
                and anomaly_score < self.anomaly_high_exit
                and smoothed_speed <= self.speed_high_exit
            )
        )
        calm_for_unstable = (
            not in_unstable_band
            or smoothed_people < self.min_people_unstable
            or (
                smoothed_density <= self.density_warn_exit
                and smoothed_entropy <= self.entropy_warn_exit
                and anomaly_score < self.anomaly_warn_exit
                and smoothed_speed <= self.speed_warn_exit
                and anomaly_trend < self.anomaly_trend_enter * 0.5
                and density_trend < self.density_trend_enter * 0.5
            )
        )

        if qualifies_high_alert:
            self._high_alert_candidate_frames += 1
            self._unstable_candidate_frames = 0
            self._mark_active()
        elif qualifies_unstable:
            self._unstable_candidate_frames += 1
            self._high_alert_candidate_frames = 0
            self._mark_active()
        else:
            self._mark_calm()

        if self._state == "HIGH_ALERT":
            if calm_for_high_alert and self._calm_frames >= self.high_alert_clear_frames:
                self._state = "UNSTABLE" if qualifies_unstable else "NORMAL"
            return self._state

        if self._high_alert_candidate_frames >= self.high_alert_confirm_frames:
            self._state = "HIGH_ALERT"
            self._calm_frames = 0
            return self._state

        if self._state == "UNSTABLE":
            if calm_for_unstable and self._calm_frames >= self.unstable_clear_frames:
                self._state = "NORMAL"
            return self._state

        if self._unstable_candidate_frames >= self.unstable_confirm_frames:
            self._state = "UNSTABLE"
            self._calm_frames = 0
            return self._state

        self._state = "NORMAL"
        return self._state
