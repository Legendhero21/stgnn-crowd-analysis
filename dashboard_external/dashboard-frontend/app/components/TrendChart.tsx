"use client";

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";
import type { TrendPoint } from "../types";

interface TrendChartProps {
    data: TrendPoint[];
    deviceId: string | null;
}

const STAMPEDE_THRESHOLD = 0.7;

export function TrendChart({ data, deviceId }: TrendChartProps) {
    if (!deviceId) {
        return (
            <section className="chart-panel">
                <div className="no-selection">
                    <p>Select a device to view trend</p>
                </div>
            </section>
        );
    }

    return (
        <section className="chart-panel">
            <h2 className="panel-title">Anomaly Score Trend — {deviceId}</h2>
            <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                        data={data}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                        <XAxis
                            dataKey="time"
                            stroke="#666"
                            tick={{ fill: "#666", fontSize: 10 }}
                            tickLine={{ stroke: "#666" }}
                        />
                        <YAxis
                            domain={[0, 1]}
                            stroke="#666"
                            tick={{ fill: "#666", fontSize: 10 }}
                            tickLine={{ stroke: "#666" }}
                            ticks={[0, 0.25, 0.5, 0.7, 1]}
                        />
                        <Tooltip
                            contentStyle={{
                                background: "#1a1a2e",
                                border: "1px solid #2a2a4a",
                                borderRadius: 8,
                                color: "#e0e0e0",
                            }}
                            labelStyle={{ color: "#a0a0a0" }}
                        />

                        {/* Stampede threshold line */}
                        <ReferenceLine
                            y={STAMPEDE_THRESHOLD}
                            stroke="#ff3366"
                            strokeDasharray="5 5"
                            label={{
                                value: "STAMPEDE",
                                position: "right",
                                fill: "#ff3366",
                                fontSize: 10,
                            }}
                        />

                        {/* Anomaly score line */}
                        <Line
                            type="monotone"
                            dataKey="anomaly_score"
                            stroke="#00ff88"
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 4, fill: "#00ff88" }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </section>
    );
}
