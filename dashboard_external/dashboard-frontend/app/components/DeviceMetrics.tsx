"use client";

import type { EdgeMetrics, AlertState } from "../types";

interface DeviceMetricsProps {
    metrics: EdgeMetrics | null;
}

function getValueClass(alertState: AlertState | undefined): string {
    switch (alertState) {
        case "HIGH_ALERT":
            return "high-alert";
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

            <div className="device-layout">
                <div className="video-panel">
                    <img
                        src={`http://127.0.0.1:8000/video/${metrics.device_id}`}
                        alt={`Live feed — ${metrics.device_id}`}
                        className="device-video"
                    />
                </div>

                <div className="metrics-grid compact">
                    <div className="metric-card">
                        <div className="metric-label">Persons</div>
                        <div className="metric-value neutral">
                            {metrics.num_persons}
                        </div>
                    </div>

                    <div className="metric-card">
                        <div className="metric-label">Density</div>
                        <div className={`metric-value ${alertClass}`}>
                            {formatNumber(metrics.crowd_density)}
                        </div>
                    </div>

                    <div className="metric-card">
                        <div className="metric-label">Anomaly</div>
                        <div className={`metric-value ${alertClass}`}>
                            {formatNumber(metrics.anomaly_score)}
                        </div>
                    </div>

                    <div className="metric-card">
                        <div className="metric-label">Alert</div>
                        <div className={`metric-value ${alertClass}`}>
                            {metrics.alert_state}
                        </div>
                    </div>

                    <div className="metric-card">
                        <div className="metric-label">Velocity</div>
                        <div className="metric-value neutral">
                            {formatNumber(metrics.avg_velocity, 3)}
                        </div>
                    </div>

                    <div className="metric-card">
                        <div className="metric-label">Latency</div>
                        <div className="metric-value neutral">
                            {formatNumber(metrics.processing_time_ms, 1)}
                            <span className="metric-unit">ms</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
