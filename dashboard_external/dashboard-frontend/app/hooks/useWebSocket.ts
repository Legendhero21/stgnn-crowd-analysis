"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { BatchUpdate, DashboardUpdate } from "../types";

const WS_URL = "ws://127.0.0.1:8000/ws/analytics";
const RECONNECT_DELAY = 3000;

interface UseWebSocketReturn {
    isConnected: boolean;
    lastMessage: BatchUpdate | null;
    connectionError: string | null;
}

export function useWebSocket(
    onMessage: (update: BatchUpdate) => void
): UseWebSocketReturn {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<BatchUpdate | null>(null);
    const [connectionError, setConnectionError] = useState<string | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const onMessageRef = useRef(onMessage);

    // Keep callback ref updated
    onMessageRef.current = onMessage;

    const connect = useCallback(() => {
        // Clean up existing connection
        if (wsRef.current) {
            wsRef.current.close();
        }

        try {
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log("[WS] Connected");
                setIsConnected(true);
                setConnectionError(null);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    // Handle initial snapshot
                    if (data.type === "initial_snapshot") {
                        // Convert snapshot to batch update format
                        const snapshot = data.data;
                        const updates: DashboardUpdate[] = [];

                        if (snapshot.server) {
                            updates.push({ type: "server_status", data: snapshot.server });
                        }
                        for (const metrics of snapshot.edge_metrics || []) {
                            updates.push({ type: "edge_metrics", data: metrics });
                        }
                        for (const status of snapshot.training_status || []) {
                            updates.push({ type: "training_status", data: status });
                        }

                        const batch: BatchUpdate = {
                            updates,
                            timestamp: snapshot.timestamp || Date.now() / 1000,
                        };
                        setLastMessage(batch);
                        onMessageRef.current(batch);
                    }
                    // Handle batch updates
                    else if (data.updates) {
                        setLastMessage(data as BatchUpdate);
                        onMessageRef.current(data as BatchUpdate);
                    }
                } catch (err) {
                    console.error("[WS] Parse error:", err);
                }
            };

            ws.onerror = (error) => {
                console.error("[WS] Error:", error);
                setConnectionError("WebSocket connection error");
            };

            ws.onclose = () => {
                console.log("[WS] Disconnected");
                setIsConnected(false);
                wsRef.current = null;

                // Schedule reconnection
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log("[WS] Reconnecting...");
                    connect();
                }, RECONNECT_DELAY);
            };
        } catch (err) {
            console.error("[WS] Connection failed:", err);
            setConnectionError("Failed to connect");

            // Retry connection
            reconnectTimeoutRef.current = setTimeout(connect, RECONNECT_DELAY);
        }
    }, []);

    useEffect(() => {
        connect();

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [connect]);

    return { isConnected, lastMessage, connectionError };
}
