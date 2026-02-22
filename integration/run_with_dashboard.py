#!/usr/bin/env python3
"""
Federated Learning + Dashboard Integration
-------------------------------------------
Runs the complete federated system with real-time dashboard observability.

Phase 6: Full real edge pipeline.
Each edge runs: Video → YOLO → GraphBuilder → TemporalBuffer → STGNN ONNX → Alert

This script:
1. Starts FederatedServer
2. Starts N FederatedClients with REAL EdgeClients (YOLO + ONNX)
3. Injects adapter into dashboard backend
4. Runs FastAPI server in background thread
5. Runs federated simulation with real edge processing

The dashboard receives REAL data from the live system.

Usage:
    python run_with_dashboard.py --num-clients 3 --rounds 5
    
    # Then open browser to http://127.0.0.1:8000/api/snapshot
    # Or connect WebSocket to ws://127.0.0.1:8000/ws/analytics
    # Video stream:  http://127.0.0.1:8000/video/edge_00
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add paths
_project_root = Path(__file__).parent.parent
_src_dir = _project_root / "src"
_backend_dir = _project_root / "dashboard_external" / "dashboard_backend"

sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_backend_dir.parent))

# Import from core system
from federated.server import FederatedServer, ServerConfig
from federated.client import FederatedClient, FederatedClientConfig, LocalTrainer
from federated.transport import LocalTransport
from federated.edge.client import EdgeClient
from federated.edge.config import create_simulation_config
from models.stgnn import STGNN

# Import dashboard components
from dashboard_backend.adapter import DashboardAdapter
from dashboard_backend import main as dashboard_main


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)





# ============================================================
# DASHBOARD SERVER THREAD
# ============================================================

def run_dashboard_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run FastAPI dashboard in background thread."""
    import uvicorn
    
    uvicorn.run(
        dashboard_main.app,
        host=host,
        port=port,
        log_level="warning",
    )


# ============================================================
# SIMULATION RUNNER
# ============================================================

class FederatedSimulation:
    """
    Runs federated learning simulation with dashboard integration.
    """
    
    def __init__(
        self,
        num_clients: int = 3,
        samples_per_client: int = 100,
        min_clients: int = 2,
        round_timeout_sec: float = 10.0,
    ):
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        
        # Server config
        self.server_config = ServerConfig(
            min_clients=min_clients,
            round_timeout_sec=round_timeout_sec,
            model_class=STGNN,
            model_kwargs={
                "in_channels": 5,
                "hidden_channels": 32,
                "out_channels": 1,
                "num_layers": 2,
            },
        )
        
        # Components
        self.server: Optional[FederatedServer] = None
        self.clients: Dict[str, FederatedClient] = {}
        self.adapter: Optional[DashboardAdapter] = None
        
        # Control
        self._stop_event = threading.Event()
        self._dashboard_thread: Optional[threading.Thread] = None
    
    def setup(self) -> None:
        """Initialize all components with REAL edge pipeline."""
        logger.info("Setting up federated simulation (REAL edge pipeline)...")
        
        # 1. Create server
        self.server = FederatedServer(self.server_config)
        
        # 2. Create clients with REAL EdgeClients
        video_path = str(
            Path(__file__).parent.parent / "data" / "videos" / "mat_dataset_full.mp4"
        )
        base_dir = str(_project_root)
        
        for i in range(self.num_clients):
            device_id = f"edge_{i:02d}"
            
            # Create real EdgeConfig via factory
            edge_config = create_simulation_config(
                video_source=video_path,
                base_dir=base_dir,
                device_id=device_id,
            )
            
            # Create REAL EdgeClient (YOLO + GraphBuilder + ONNX + Alert)
            real_edge = EdgeClient(edge_config)
            
            # Create transport
            transport = LocalTransport(self.server)
            
            # Create federated client config
            client_config = FederatedClientConfig(
                training_interval_sec=5.0,
                heartbeat_interval_sec=10.0,
                max_local_epochs=2,
                min_samples_for_training=16,
                learning_rate=0.001,
                batch_size=8,
            )
            
            # Create local trainer
            trainer = LocalTrainer(
                model_class=STGNN,
                model_kwargs=self.server_config.model_kwargs,
                learning_rate=client_config.learning_rate,
            )
            
            # Create federated client
            client = FederatedClient(
                edge_client=real_edge,
                transport=transport,
                trainer=trainer,
                config=client_config,
            )
            
            self.clients[device_id] = client
            logger.info("[%s] Real EdgeClient created", device_id)
        
        # 3. Create dashboard adapter
        self.adapter = DashboardAdapter(
            server=self.server,
            clients=self.clients,
        )
        
        # 4. Inject adapter into dashboard backend
        dashboard_main.set_adapter(self.adapter)
        
        logger.info(
            "Simulation setup complete: %d real edge clients",
            len(self.clients),
        )
    
    def start_dashboard(self, port: int = 8000) -> None:
        """Start dashboard server in background."""
        self._dashboard_thread = threading.Thread(
            target=run_dashboard_server,
            kwargs={"port": port},
            daemon=True,
        )
        self._dashboard_thread.start()
        logger.info("Dashboard started at http://127.0.0.1:%d", port)
    
    def run_simulation(self, num_rounds: int = 3) -> None:
        """
        Run federated learning simulation with real edge pipeline.
        
        EdgeClient.start() launches background threads that process
        video → YOLO → Graph → STGNN → Alert → TrainingBuffer.
        We then wait for enough training samples to accumulate
        before triggering training cycles.
        
        Args:
            num_rounds: Number of federated rounds to run.
        """
        logger.info("Starting simulation (%d rounds)...", num_rounds)
        
        try:
            # Start all clients (this initializes + registers + starts edge processing)
            for device_id, client in self.clients.items():
                success = client.start(blocking=False)
                if success:
                    logger.info("[%s] Started (real pipeline running)", device_id)
                else:
                    logger.error("[%s] Failed to start!", device_id)
            
            # Run rounds
            for round_num in range(num_rounds):
                logger.info("=" * 50)
                logger.info("ROUND %d / %d", round_num + 1, num_rounds)
                logger.info("=" * 50)
                
                # Wait for real training samples to accumulate
                # EdgeClient processes video frames in background and populates TrainingBuffer
                min_samples = 16
                max_wait = 60  # seconds
                logger.info("Waiting for training samples to accumulate (min=%d)...", min_samples)
                
                wait_start = time.time()
                while time.time() - wait_start < max_wait:
                    all_ready = True
                    for device_id, client in self.clients.items():
                        buf = client._edge_client.training_buffer
                        sample_count = len(buf) if buf else 0
                        if sample_count < min_samples:
                            all_ready = False
                    
                    if all_ready:
                        break
                    time.sleep(2.0)
                
                # Log buffer status
                for device_id, client in self.clients.items():
                    buf = client._edge_client.training_buffer
                    count = len(buf) if buf else 0
                    edge_stats = client._edge_client.get_stats()
                    logger.info(
                        "[%s] frames=%d, buffer=%d samples",
                        device_id,
                        edge_stats.get("frame_count", 0),
                        count,
                    )
                
                # Sync models before training
                for device_id, client in self.clients.items():
                    client._poll_for_model()
                
                # Training cycle for each client
                for device_id, client in self.clients.items():
                    client._training_cycle()
                    stats = client.get_stats()
                    logger.info(
                        "[%s] Training complete, total_samples_trained=%d",
                        device_id,
                        stats.get("samples_trained", 0),
                    )
                
                # Wait for aggregation
                time.sleep(1.0)
                
                # Poll for new aggregated model
                for device_id, client in self.clients.items():
                    client._poll_for_model()
                
                # Log server stats
                server_stats = self.server.get_stats()
                logger.info(
                    "Server: version=%d, round=%d, status=%s",
                    server_stats["model_version"],
                    server_stats["round_id"],
                    server_stats["round_status"],
                )
                
                # Small delay between rounds
                time.sleep(0.5)
            
            logger.info("=" * 50)
            logger.info("SIMULATION COMPLETE")
            logger.info("=" * 50)
            
        except KeyboardInterrupt:
            logger.info("Simulation interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        for client in self.clients.values():
            try:
                client.stop()
            except Exception:
                pass
        
        if self.server:
            try:
                self.server.shutdown()
            except Exception:
                pass
        
        logger.info("Simulation cleanup complete")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run federated learning with dashboard",
    )
    parser.add_argument(
        "--num-clients", "-n",
        type=int,
        default=3,
        help="Number of edge clients",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=5,
        help="Number of federated rounds",
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=100,
        help="Samples per client",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Dashboard port",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Keep dashboard running after simulation",
    )
    
    args = parser.parse_args()
    
    # Create and run simulation
    sim = FederatedSimulation(
        num_clients=args.num_clients,
        samples_per_client=args.samples,
    )
    
    sim.setup()
    sim.start_dashboard(port=args.port)
    
    # Give dashboard time to start
    time.sleep(1.0)
    
    print("\n" + "=" * 60)
    print("Dashboard running at: http://127.0.0.1:%d" % args.port)
    print("REST API: http://127.0.0.1:%d/api/snapshot" % args.port)
    print("WebSocket: ws://127.0.0.1:%d/ws/analytics" % args.port)
    print("=" * 60 + "\n")
    
    sim.run_simulation(num_rounds=args.rounds)
    
    if args.dashboard_only:
        print("\nDashboard still running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
