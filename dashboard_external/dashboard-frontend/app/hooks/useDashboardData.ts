"use client";

import { useState, useCallback, useEffect } from "react";
import type {
    DashboardState,
    BatchUpdate,
    EdgeMetrics,
    TrainingStatus,
    ServerStatus,
    Snapshot,
    TrendPoint,
} from "../types";

const API_BASE = "http://127.0.0.1:8000";
const MAX_TREND_POINTS = 20;

interface UseDashboardDataReturn {
    state: DashboardState;
    trendData: Map<string, TrendPoint[]>;
    selectDevice: (deviceId: string) => void;
    handleBatchUpdate: (batch: BatchUpdate) => void;
    fetchInitialData: () => Promise<void>;
}

export function useDashboardData(): UseDashboardDataReturn {
    const [state, setState] = useState<DashboardState>({
        server: null,
        edgeMetrics: new Map(),
        trainingStatus: new Map(),
        deviceIds: [],
        selectedDevice: null,
        isConnected: false,
        lastUpdate: 0,
    });

    const [trendData, setTrendData] = useState<Map<string, TrendPoint[]>>(new Map());

    // Fetch initial snapshot from REST API
    const fetchInitialData = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/snapshot`);
            if (!response.ok) {
                console.error("[API] Snapshot fetch failed:", response.status);
                return;
            }

            const snapshot: Snapshot = await response.json();

            setState((prev) => {
                const newEdgeMetrics = new Map(prev.edgeMetrics);
                const newTrainingStatus = new Map(prev.trainingStatus);
                const deviceIds = new Set(prev.deviceIds);

                // Process edge metrics
                for (const metrics of snapshot.edge_metrics || []) {
                    newEdgeMetrics.set(metrics.device_id, metrics);
                    deviceIds.add(metrics.device_id);
                }

                // Process training status
                for (const status of snapshot.training_status || []) {
                    newTrainingStatus.set(status.device_id, status);
                    deviceIds.add(status.device_id);
                }

                const deviceIdArray = Array.from(deviceIds).sort();

                return {
                    ...prev,
                    server: snapshot.server,
                    edgeMetrics: newEdgeMetrics,
                    trainingStatus: newTrainingStatus,
                    deviceIds: deviceIdArray,
                    selectedDevice: prev.selectedDevice || deviceIdArray[0] || null,
                    lastUpdate: snapshot.timestamp || Date.now() / 1000,
                };
            });

            console.log("[API] Initial snapshot loaded");
        } catch (err) {
            console.error("[API] Fetch error:", err);
        }
    }, []);

    // Handle incoming WebSocket batch updates
    const handleBatchUpdate = useCallback((batch: BatchUpdate) => {
        setState((prev) => {
            const newEdgeMetrics = new Map(prev.edgeMetrics);
            const newTrainingStatus = new Map(prev.trainingStatus);
            const deviceIds = new Set(prev.deviceIds);
            let newServer = prev.server;

            for (const update of batch.updates) {
                switch (update.type) {
                    case "edge_metrics": {
                        const metrics = update.data as EdgeMetrics;
                        newEdgeMetrics.set(metrics.device_id, metrics);
                        deviceIds.add(metrics.device_id);
                        break;
                    }
                    case "training_status": {
                        const status = update.data as TrainingStatus;
                        newTrainingStatus.set(status.device_id, status);
                        deviceIds.add(status.device_id);
                        break;
                    }
                    case "server_status": {
                        newServer = update.data as ServerStatus;
                        break;
                    }
                }
            }

            const deviceIdArray = Array.from(deviceIds).sort();

            return {
                ...prev,
                server: newServer,
                edgeMetrics: newEdgeMetrics,
                trainingStatus: newTrainingStatus,
                deviceIds: deviceIdArray,
                selectedDevice: prev.selectedDevice || deviceIdArray[0] || null,
                lastUpdate: batch.timestamp,
            };
        });

        // Update trend data for each device with new edge metrics
        setTrendData((prev) => {
            const newTrendData = new Map(prev);

            for (const update of batch.updates) {
                if (update.type === "edge_metrics") {
                    const metrics = update.data as EdgeMetrics;
                    const deviceTrend = newTrendData.get(metrics.device_id) || [];

                    const newPoint: TrendPoint = {
                        time: new Date(metrics.timestamp_ms).toLocaleTimeString(),
                        timestamp: metrics.timestamp_ms,
                        anomaly_score: metrics.anomaly_score,
                        crowd_density: metrics.crowd_density,
                    };

                    const updated = [...deviceTrend, newPoint].slice(-MAX_TREND_POINTS);
                    newTrendData.set(metrics.device_id, updated);
                }
            }

            return newTrendData;
        });
    }, []);

    // Select a device
    const selectDevice = useCallback((deviceId: string) => {
        setState((prev) => ({
            ...prev,
            selectedDevice: deviceId,
        }));
    }, []);

    // Fetch initial data on mount
    useEffect(() => {
        fetchInitialData();
    }, [fetchInitialData]);

    return {
        state,
        trendData,
        selectDevice,
        handleBatchUpdate,
        fetchInitialData,
    };
}
