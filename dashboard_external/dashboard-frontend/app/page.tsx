"use client";

import { useCallback } from "react";

import { useWebSocket } from "./hooks/useWebSocket";
import { useDashboardData } from "./hooks/useDashboardData";

import { TopBar } from "./components/TopBar";
import { DeviceList } from "./components/DeviceList";
import { DeviceMetrics } from "./components/DeviceMetrics";
import { TrainingStatus } from "./components/TrainingStatus";
import { TrendChart } from "./components/TrendChart";
import { AlertBanner } from "./components/AlertBanner";

export default function Dashboard() {
  const {
    state,
    trendData,
    selectDevice,
    handleBatchUpdate,
  } = useDashboardData();

  // Handle WebSocket messages
  const onMessage = useCallback(
    (batch: Parameters<typeof handleBatchUpdate>[0]) => {
      handleBatchUpdate(batch);
    },
    [handleBatchUpdate]
  );

  const { isConnected } = useWebSocket(onMessage);

  // Get data for selected device
  const selectedMetrics = state.selectedDevice
    ? state.edgeMetrics.get(state.selectedDevice) ?? null
    : null;

  const selectedTraining = state.selectedDevice
    ? state.trainingStatus.get(state.selectedDevice) ?? null
    : null;

  const selectedTrend = state.selectedDevice
    ? trendData.get(state.selectedDevice) ?? []
    : [];

  return (
    <div className="dashboard">
      {/* Global stampede alert */}
      <AlertBanner edgeMetrics={state.edgeMetrics} />

      {/* Top bar with logo and server info */}
      <TopBar server={state.server} isConnected={isConnected} />

      {/* Left sidebar - Device list */}
      <DeviceList
        deviceIds={state.deviceIds}
        selectedDevice={state.selectedDevice}
        edgeMetrics={state.edgeMetrics}
        onSelect={selectDevice}
      />

      {/* Center panel - Device metrics */}
      <DeviceMetrics metrics={selectedMetrics} />

      {/* Right panel - Training status */}
      <TrainingStatus status={selectedTraining} />

      {/* Bottom panel - Trend chart */}
      <TrendChart data={selectedTrend} deviceId={state.selectedDevice} />
    </div>
  );
}
