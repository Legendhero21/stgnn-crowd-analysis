"use client";

import type { TrainingStatus as TrainingStatusType } from "../types";

interface TrainingStatusProps {
    status: TrainingStatusType | null;
}

function formatNumber(value: number): string {
    if (value >= 1000000) {
        return (value / 1000000).toFixed(1) + "M";
    }
    if (value >= 1000) {
        return (value / 1000).toFixed(1) + "K";
    }
    return value.toString();
}

export function TrainingStatus({ status }: TrainingStatusProps) {
    if (!status) {
        return (
            <aside className="training-panel">
                <h2 className="panel-title">Training Status</h2>
                <div className="no-selection">
                    <p>No training data</p>
                </div>
            </aside>
        );
    }

    return (
        <aside className="training-panel">
            <h2 className="panel-title">Training Status</h2>

            <div className="training-stat">
                <div className="training-label">State</div>
                <div className="training-value state">{status.state}</div>
            </div>

            <div className="training-stat">
                <div className="training-label">Registered</div>
                <div className="training-value">
                    {status.is_registered ? "Yes" : "No"}
                </div>
            </div>

            <div className="training-stat">
                <div className="training-label">Training Rounds</div>
                <div className="training-value">{status.training_rounds}</div>
            </div>

            <div className="training-stat">
                <div className="training-label">Samples Trained</div>
                <div className="training-value">
                    {formatNumber(status.samples_trained)}
                </div>
            </div>

            <div className="training-stat">
                <div className="training-label">Samples Buffered</div>
                <div className="training-value">{status.samples_buffered}</div>
            </div>

            <div className="training-stat">
                <div className="training-label">Model Version</div>
                <div className="training-value">v{status.model_version}</div>
            </div>
        </aside>
    );
}
