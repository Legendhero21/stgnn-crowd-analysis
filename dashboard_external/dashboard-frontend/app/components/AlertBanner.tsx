"use client";

import type { EdgeMetrics } from "../types";

interface AlertBannerProps {
    edgeMetrics: Map<string, EdgeMetrics>;
}

export function AlertBanner({ edgeMetrics }: AlertBannerProps) {
    let hasHighAlert = false;
    const highAlertDevices: string[] = [];

    edgeMetrics.forEach((metrics, deviceId) => {
        if (metrics.alert_state === "HIGH_ALERT") {
            hasHighAlert = true;
            highAlertDevices.push(deviceId);
        }
    });

    if (!hasHighAlert) {
        return null;
    }

    return (
        <div className="alert-banner">
            HIGH ALERT — {highAlertDevices.join(", ")}
        </div>
    );
}
