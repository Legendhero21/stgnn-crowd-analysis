"use client";

import type { EdgeMetrics, AlertState } from "../types";

interface DeviceListProps {
    deviceIds: string[];
    selectedDevice: string | null;
    edgeMetrics: Map<string, EdgeMetrics>;
    onSelect: (deviceId: string) => void;
}

function getAlertClass(state: AlertState): string {
    switch (state) {
        case "STAMPEDE":
            return "stampede";
        case "UNSTABLE":
            return "unstable";
        default:
            return "normal";
    }
}

export function DeviceList({
    deviceIds,
    selectedDevice,
    edgeMetrics,
    onSelect,
}: DeviceListProps) {
    if (deviceIds.length === 0) {
        return (
            <aside className="sidebar">
                <h2 className="sidebar-title">Devices</h2>
                <div className="no-selection">
                    <p>No devices connected</p>
                </div>
            </aside>
        );
    }

    return (
        <aside className="sidebar">
            <h2 className="sidebar-title">Devices ({deviceIds.length})</h2>
            <div className="device-list">
                {deviceIds.map((deviceId) => {
                    const metrics = edgeMetrics.get(deviceId);
                    const alertState = metrics?.alert_state || "NORMAL";
                    const modelVersion = metrics?.model_version ?? 0;

                    return (
                        <div
                            key={deviceId}
                            className={`device-item ${selectedDevice === deviceId ? "selected" : ""}`}
                            onClick={() => onSelect(deviceId)}
                        >
                            <div className="device-id">{deviceId}</div>
                            <div className="device-meta">
                                <span className="device-version">v{modelVersion}</span>
                                <span className={`alert-badge ${getAlertClass(alertState)}`}>
                                    {alertState}
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </aside>
    );
}
