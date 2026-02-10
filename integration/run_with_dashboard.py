#!/usr/bin/env python3
"""
Federated Learning + Dashboard Integration
-------------------------------------------
Runs the complete federated system with real-time dashboard observability.

This script:
1. Starts FederatedServer
2. Starts N FederatedClients (with mock EdgeClients)
3. Injects adapter into dashboard backend
4. Runs FastAPI server in background thread
5. Runs federated simulation

The dashboard receives REAL data from the live system.

Usage:
    python run_with_dashboard.py --num-clients 3 --rounds 5
    
    # Then open browser to http://127.0.0.1:8000/api/snapshot
    # Or connect WebSocket to ws://127.0.0.1:8000/ws/analytics
"""

from __future__ import annotations
import cv2
import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add paths
_project_root = Path(__file__).parent.parent
_src_dir = _project_root / "src"
_backend_dir = _project_root / "dashboard_external" / "dashboard_backend"

sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_backend_dir.parent))

import numpy as np
import torch

# Import from core system (LOCKED Phase 1-4)
from federated.server import FederatedServer, ServerConfig
from federated.client import FederatedClient, FederatedClientConfig, LocalTrainer
from federated.transport import LocalTransport
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
# MOCK EDGE CLIENT (for simulation without video)
# ============================================================

@dataclass  
class MockFrameResult:
    """Simulated frame result."""
    frame_idx: int
    timestamp_ms: float
    centers: List[tuple]
    num_persons: int
    graph: Optional[dict]
    anomaly_score: float
    metrics: Dict[str, float]
    alert_state: str
    model_version: int
    processing_time_ms: float


class MockTrainingBuffer:
    """Simulated training buffer with synthetic data."""
    
    def __init__(self, num_samples: int = 100):
        self._samples = []
        self._generate_samples(num_samples)
    

    def _generate_samples(self, n: int) -> None:
        for _ in range(n):
            # T=5, N=10-30 random nodes, F=5 features
            num_nodes = np.random.randint(10, 30)
            x_seq = np.random.randn(5, num_nodes, 5).astype(np.float32)
            y = np.random.randn(num_nodes, 1).astype(np.float32)
            edge_index = np.random.randint(0, num_nodes, (2, num_nodes * 2))
            self._samples.append((x_seq, y, edge_index))
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def get_batch(self, batch_size: int) -> Optional[tuple]:
        if len(self._samples) < batch_size:
            return None
        batch = self._samples[:batch_size]
        self._samples = self._samples[batch_size:]
        return batch
    
    def iter_batches(self, batch_size: int):
        """Yield batches for training in (x_seq, edge_index, target) format."""
        for i in range(0, len(self._samples), batch_size):
            batch = self._samples[i:i + batch_size]
            if not batch:
                break
            # Combine batch into tensors
            # Shape: [B, T, N, F] but trainer expects [1, T, N, F]
            # For simplicity, yield one sample at a time as [1, T, N, F]
            for x_seq, target, edge_index in batch:
                x_tensor = torch.tensor(x_seq).unsqueeze(0)  # [1, T, N, F]
                edge_tensor = torch.tensor(edge_index, dtype=torch.long)
                target_tensor = torch.tensor(target).unsqueeze(0)  # [1, N, out_dim]
                yield x_tensor, edge_tensor, target_tensor
    
    def clear(self) -> None:
        self._samples.clear()
    
    def get_stats(self) -> dict:
        return {"size": len(self._samples), "capacity": 1000}


class MockEdgeClient:
    """
    Simulated EdgeClient for integration testing.
    
    Generates fake metrics without video/YOLO/ONNX.
    """
    
    def __init__(self, device_id: str,video_path: Path, num_samples: int = 100):
        self._device_id = device_id
        self._is_initialized = False
        self._is_running = False
        self._training_buffer = MockTrainingBuffer(num_samples)
        self._model_version = 0
        self._frame_idx = 0
        self._latest_result: Optional[MockFrameResult] = None
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._video_path = video_path
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
    
    @property
    def device_id(self) -> str:
        return self._device_id
    
    @property
    def training_buffer(self):
        return self._training_buffer
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    @property
    def model_version(self) -> int:
        return self._model_version
    
    def initialize(self) -> bool:
        self._is_initialized = True
        self._start_time = time.time()
        logger.info("[%s] MockEdgeClient initialized", self._device_id)
        return True
    
    def start(self, blocking: bool = False) -> bool:
        self._is_running = True
        return True
    
    def stop(self, timeout: float = 5.0) -> None:
        self._is_running = False
        if self._cap:
            self._cap.release()
    
    def update_model(self, new_onnx_path: str, new_version: int) -> bool:
        self._model_version = new_version
        logger.info("[%s] Model updated to v%d", self._device_id, new_version)
        return True
    
    def replace_training_buffer(self, new_buffer) -> None:
        self._training_buffer = new_buffer
    
    def simulate_frame(self) -> None:
        """Read next video frame and generate simulated analytics."""
        with self._lock:
            ret, frame = self._cap.read()

        # Loop video when finished
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if not ret:
                return

        self._frame_idx += 1

        h, w, _ = frame.shape
        num_persons = np.random.randint(5, 50)

        self._latest_result = MockFrameResult(
            frame_idx=self._frame_idx,
            timestamp_ms=time.time() * 1000,
            centers=[
                (np.random.randint(0, w), np.random.randint(0, h))
                for _ in range(num_persons)
            ],
            num_persons=num_persons,
            graph=None,
            anomaly_score=np.random.random() * 0.6,
            metrics={
                "density": min(num_persons / 50.0, 1.0),
                "avg_velocity": np.random.random() * 2.0,
                "flow_magnitude": np.random.random() * 5.0,
            },
            alert_state=np.random.choice(
                ["NORMAL", "UNSTABLE", "STAMPEDE"],
                p=[0.7, 0.2, 0.1],
            ),
            model_version=self._model_version,
            processing_time_ms=np.random.random() * 40,
        )

    
    def get_latest_result(self) -> Optional[MockFrameResult]:
        with self._lock:
            return self._latest_result
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._start_time if self._start_time else 0
            return {
                "device_id": self._device_id,
                "is_running": self._is_running,
                "frame_count": self._frame_idx,
                "elapsed_sec": elapsed,
                "fps": self._frame_idx / elapsed if elapsed > 0 else 0,
                "model_version": self._model_version,
                "training_buffer": self._training_buffer.get_stats(),
            }


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
        """Initialize all components."""
        logger.info("Setting up federated simulation...")
        
        # 1. Create server
        self.server = FederatedServer(self.server_config)
        
        # 2. Create clients with mock edge devices
        for i in range(self.num_clients):
            device_id = f"edge_{i:02d}"
            
            # Create mock edge client
            video_path = (
                Path(__file__).parent.parent
                / "data"
                / "videos"
                / "mat_dataset_full.mp4"
            )

            mock_edge = MockEdgeClient(
                device_id=device_id,
                video_path=video_path,
                num_samples=self.samples_per_client,
            )
            mock_edge.initialize()
            
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
                edge_client=mock_edge,
                transport=transport,
                trainer=trainer,
                config=client_config,
            )
            
            self.clients[device_id] = client
        
        # 3. Create dashboard adapter
        self.adapter = DashboardAdapter(
            server=self.server,
            clients=self.clients,
        )
        
        # 4. Inject adapter into dashboard backend
        dashboard_main.set_adapter(self.adapter)
        
        logger.info(
            "Simulation setup complete: %d clients",
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
        Run federated learning simulation.
        
        Args:
            num_rounds: Number of federated rounds to run.
        """
        logger.info("Starting simulation (%d rounds)...", num_rounds)
        
        try:
            # Register all clients
            for device_id, client in self.clients.items():
                client.start()
                # Simulate initial registration
                client._register()
                logger.info("[%s] Registered", device_id)
            
            # Run rounds
            for round_num in range(num_rounds):
                logger.info("=" * 50)
                logger.info("ROUND %d", round_num + 1)
                logger.info("=" * 50)
                
                # Simulate frames for each client
                for device_id, client in self.clients.items():
                    mock_edge = client._edge_client
                    for _ in range(10):
                        mock_edge.simulate_frame()
                    
                    # Refill training buffer if needed
                    if len(mock_edge.training_buffer) < 32:
                        mock_edge.replace_training_buffer(
                            MockTrainingBuffer(self.samples_per_client)
                        )
                
                # Training cycle for each client
                for device_id, client in self.clients.items():
                    client._training_cycle()
                    logger.info(
                        "[%s] Training complete, samples=%d",
                        device_id,
                        client.get_stats()["samples_trained"],
                    )
                
                # Wait for aggregation
                time.sleep(1.0)
                
                # Poll for new models
                for device_id, client in self.clients.items():
                    client._poll_for_model()
                
                # Small delay between rounds
                time.sleep(0.5)
                
                # Log server stats
                server_stats = self.server.get_stats()
                logger.info(
                    "Server: version=%d, round=%d, status=%s",
                    server_stats["model_version"],
                    server_stats["round_id"],
                    server_stats["round_status"],
                )
            
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
