"use client";

import type { ServerStatus } from "../types";

interface TopBarProps {
    server: ServerStatus | null;
    isConnected: boolean;
}

export function TopBar({ server, isConnected }: TopBarProps) {
    return (
        <header className="topbar">
            <div className="topbar-left">
                <div className="logo-placeholder">vvv</div>
                <h1 className="project-name">vvv</h1>
            </div>

            <div className="topbar-right">
                <div className="connection-status">
                    <span
                        className={`status-dot ${isConnected ? "connected" : "disconnected"}`}
                    />
                    <span>{isConnected ? "Live" : "Connecting..."}</span>
                </div>

                {server && (
                    <>
                        <div className="server-stat">
                            <span className="server-stat-label">Model Version</span>
                            <span className="server-stat-value">v{server.model_version}</span>
                        </div>
                        <div className="server-stat">
                            <span className="server-stat-label">Round</span>
                            <span className="server-stat-value">#{server.round_id}</span>
                        </div>
                        <div className="server-stat">
                            <span className="server-stat-label">Status</span>
                            <span className="server-stat-value">{server.round_status}</span>
                        </div>
                    </>
                )}
            </div>
        </header>
    );
}
