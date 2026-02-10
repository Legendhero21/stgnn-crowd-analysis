"use client";

import type { EdgeMetrics } from "../types";

interface AlertBannerProps {
    edgeMetrics: Map<string, EdgeMetrics>;
}

export function AlertBanner({ edgeMetrics }: AlertBannerProps) {
    // Check if ANY device has STAMPEDE alert
    let hasStampede = false;
    const stampedeDevices: string[] = [];

    edgeMetrics.forEach((metrics, deviceId) => {
        if (metrics.alert_state === "STAMPEDE") {
            hasStampede = true;
            stampedeDevices.push(deviceId);
        }
    });

    if (!hasStampede) {
        return null;
    }

    return (
        <div className="alert-banner">
            🚨 STAMPEDE ALERT — {stampedeDevices.join(", ")}
        </div>
    );
}
