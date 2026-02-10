"use client";

import type { EdgeMetrics, AlertState } from "../types";

interface DeviceMetricsProps {
    metrics: EdgeMetrics | null;
}

function getValueClass(alertState: AlertState | undefined): string {
    switch (alertState) {
        case "STAMPEDE":
            return "stampede";
        case "UNSTABLE":
            return "unstable";
        default:
            return "normal";
    }
}

function formatNumber(value: number, decimals: number = 2): string {
    return value.toFixed(decimals);
}

export function DeviceMetrics({ metrics }: DeviceMetricsProps) {
    if (!metrics) {
        return (
            <section className="main-panel">
                <div className="no-selection">
                    <div className="no-selection-icon">📊</div>
                    <p>Select a device to view metrics</p>
                </div>
            </section>
        );
    }

    const alertClass = getValueClass(metrics.alert_state);

    return (
        <section className="main-panel">
            <h2 className="panel-title">
                Device Metrics — {metrics.device_id}
            </h2>

            <div className="metrics-grid">
                {/* Number of Persons */}
                <div className="metric-card">
                    <div className="metric-label">Persons Detected</div>
                    <div className="metric-value neutral">
                        {metrics.num_persons}
                        <span className="metric-unit">people</span>
                    </div>
                </div>

                {/* Crowd Density */}
                <div className="metric-card">
                    <div className="metric-label">Crowd Density</div>
                    <div className={`metric-value ${alertClass}`}>
                        {formatNumber(metrics.crowd_density)}
                    </div>
                </div>

                {/* Anomaly Score */}
                <div className="metric-card">
                    <div className="metric-label">Anomaly Score</div>
                    <div className={`metric-value ${alertClass}`}>
                        {formatNumber(metrics.anomaly_score)}
                    </div>
                </div>

                {/* Alert State */}
                <div className="metric-card">
                    <div className="metric-label">Alert State</div>
                    <div className={`metric-value ${alertClass}`}>
                        {metrics.alert_state}
                    </div>
                </div>

                {/* Model Version */}
                <div className="metric-card">
                    <div className="metric-label">Model Version</div>
                    <div className="metric-value neutral">
                        v{metrics.model_version}
                    </div>
                </div>

                {/* Processing Time */}
                <div className="metric-card">
                    <div className="metric-label">Processing Time</div>
                    <div className="metric-value neutral">
                        {formatNumber(metrics.processing_time_ms, 1)}
                        <span className="metric-unit">ms</span>
                    </div>
                </div>
            </div>
        </section>
    );
}
