// Dashboard TypeScript Types
// Aligned with backend schemas.py

// Enums
export type AlertState = "NORMAL" | "UNSTABLE" | "STAMPEDE";
export type ClientState = "IDLE" | "COLLECTING" | "TRAINING" | "SUBMITTING" | "WAITING_MODEL";
export type RoundStatus = "WAITING" | "AGGREGATING" | "DISTRIBUTING" | "COMPLETE";
export type DeviceStatus = "ACTIVE" | "STALE" | "OFFLINE";

// Edge Metrics (from EdgeClient.get_latest_result)
export interface EdgeMetrics {
    device_id: string;
    timestamp_ms: number;
    frame_idx: number;
    num_persons: number;
    anomaly_score: number;
    alert_state: AlertState;
    model_version: number;
    crowd_density: number;
    avg_velocity: number;
    flow_magnitude: number;
    processing_time_ms: number;
}

// Training Status (from FederatedClient.get_stats)
export interface TrainingStatus {
    device_id: string;
    state: ClientState;
    model_version: number;
    is_registered: boolean;
    training_rounds: number;
    samples_trained: number;
    samples_buffered: number;
    last_training_time: number | null;
}

// Registry Stats (nested in ServerStatus)
export interface RegistryStats {
    total_devices: number;
    active_devices: number;
    stale_devices: number;
    stale_timeout_sec: number;
}

// Model Info (nested in ServerStatus)
export interface ModelInfo {
    param_count: number;
    onnx_path: string | null;
}

// Server Status (from FederatedServer.get_stats)
export interface ServerStatus {
    round_id: number;
    round_status: RoundStatus;
    model_version: number;
    pending_updates: number;
    pending_distributions: number;
    registry: RegistryStats;
    model_info: ModelInfo;
}

// Device Info (from DeviceRegistry)
export interface DeviceInfo {
    device_id: string;
    status: DeviceStatus;
    model_version: number;
    last_seen: number;
    registered_at: number;
}

// WebSocket message types
export interface DashboardUpdate {
    type: "edge_metrics" | "training_status" | "server_status" | "device_list";
    data: EdgeMetrics | TrainingStatus | ServerStatus | DeviceInfo[];
}

export interface BatchUpdate {
    updates: DashboardUpdate[];
    timestamp: number;
}

// Initial snapshot from /api/snapshot
export interface Snapshot {
    server: ServerStatus | null;
    devices: DeviceInfo[];
    edge_metrics: EdgeMetrics[];
    training_status: TrainingStatus[];
    timestamp: number;
}

// Dashboard state
export interface DashboardState {
    server: ServerStatus | null;
    edgeMetrics: Map<string, EdgeMetrics>;
    trainingStatus: Map<string, TrainingStatus>;
    deviceIds: string[];
    selectedDevice: string | null;
    isConnected: boolean;
    lastUpdate: number;
}

// Trend data point for charts
export interface TrendPoint {
    time: string;
    timestamp: number;
    anomaly_score: number;
    crowd_density: number;
}
