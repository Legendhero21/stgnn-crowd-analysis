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

            {/* Side-by-side: video left, metrics right */}
            <div style={{
                display: "flex",
                gap: "1rem",
                alignItems: "flex-start",
            }}>
                {/* Video stream — compact */}
                <div style={{
                    flex: "0 0 55%",
                    borderRadius: "8px",
                    overflow: "hidden",
                    border: "1px solid var(--border-color, #333)",
                    backgroundColor: "#000",
                }}>
                    <img
                        src={`http://127.0.0.1:8000/video/${metrics.device_id}`}
                        alt={`Live feed — ${metrics.device_id}`}
                        style={{
                            width: "100%",
                            height: "auto",
                            display: "block",
                        }}
                    />
                </div>

                {/* Metrics grid — right side */}
                <div style={{
                    flex: "1",
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "0.6rem",
                }}>
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
                        <div className="metric-label">Model</div>
                        <div className="metric-value neutral">
                            v{metrics.model_version}
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
