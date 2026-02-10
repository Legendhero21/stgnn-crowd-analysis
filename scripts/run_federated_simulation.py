#!/usr/bin/env python3
"""
Federated Learning Simulation
-----------------------------
End-to-end simulation of federated STGNN training.

This script demonstrates:
- 1 FederatedServer
- N FederatedClients (each wrapping its own EdgeClient)
- Complete federated rounds: registration → training → aggregation → model sync

Usage:
    python scripts/run_federated_simulation.py --help
    python scripts/run_federated_simulation.py --num-clients 3 --video data/test.mp4
    python scripts/run_federated_simulation.py --num-clients 2 --simulation-rounds 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Optional

# Add src to path
_script_dir = Path(__file__).parent
_src_dir = _script_dir.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("federated_simulation")


# ============================================================
# Dummy STGNN Model (for simulation without full dependencies)
# ============================================================

class DummySTGNN(nn.Module):
    """
    Minimal STGNN-like model for simulation.
    
    In production, replace with the actual STGNN model.
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Simple MLP-based approximation
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, N, F]
            edge_index: Edge indices [2, E]
        
        Returns:
            Predictions [B, N, out_channels]
        """
        # Take last timestep
        x = x[:, -1, :, :]  # [B, N, F]
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# ============================================================
# Synthetic Data Generator
# ============================================================

class SyntheticTrainingBuffer:
    """
    Synthetic training buffer for simulation.
    
    Generates random training samples without needing video/YOLO.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        temporal_window: int = 10,
        num_nodes: int = 20,
        num_features: int = 5,
    ):
        self.num_samples = num_samples
        self.temporal_window = temporal_window
        self.num_nodes = num_nodes
        self.num_features = num_features
        self._samples = []
        
        # Generate synthetic samples
        for _ in range(num_samples):
            x_seq = torch.randn(1, temporal_window, num_nodes, num_features)
            edge_index = self._random_edges(num_nodes, num_edges=num_nodes * 2)
            target = torch.randn(1, num_nodes, 2)  # Predict x, y
            self._samples.append((x_seq, edge_index, target))
    
    def _random_edges(self, num_nodes: int, num_edges: int) -> torch.Tensor:
        """Generate random edges."""
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        return torch.stack([src, dst], dim=0)
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def iter_batches(self, batch_size: int):
        """Iterate over batches."""
        for i in range(0, len(self._samples), batch_size):
            batch = self._samples[i:i + batch_size]
            x_seqs = torch.cat([s[0] for s in batch], dim=0)
            edge_index = batch[0][1]  # Use same edge structure
            targets = torch.cat([s[2] for s in batch], dim=0)
            yield x_seqs, edge_index, targets
    
    def clear(self):
        """Clear buffer."""
        self._samples = []


# ============================================================
# Mock EdgeClient for Simulation
# ============================================================

class MockEdgeClient:
    """
    Mock EdgeClient for simulation without video/YOLO dependencies.
    """
    
    def __init__(self, device_id: str, num_samples: int = 100):
        self._device_id = device_id
        self._is_initialized = False
        self._is_running = False
        self._training_buffer = SyntheticTrainingBuffer(num_samples=num_samples)
        self._model_version = 0
    
    @property
    def device_id(self) -> str:
        return self._device_id
    
    @property
    def training_buffer(self):
        return self._training_buffer
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def initialize(self) -> bool:
        self._is_initialized = True
        logger.info("[%s] MockEdgeClient initialized", self._device_id)
        return True
    
    def start(self, blocking: bool = False) -> None:
        self._is_running = True
        logger.info("[%s] MockEdgeClient started", self._device_id)
    
    def stop(self, timeout: float = 5.0) -> None:
        self._is_running = False
        logger.info("[%s] MockEdgeClient stopped", self._device_id)
    
    def update_onnx_model(self, new_model_path: str, new_version: int) -> None:
        """Mock ONNX update."""
        self._model_version = new_version
        logger.info("[%s] ONNX updated to v%d", self._device_id, new_version)
    
    def replace_training_buffer(self, new_buffer) -> None:
        """Replace training buffer."""
        self._training_buffer = new_buffer


# ============================================================
# Simulation Runner
# ============================================================

class FederatedSimulation:
    """
    Runs federated learning simulation.
    """
    
    def __init__(
        self,
        num_clients: int = 3,
        samples_per_client: int = 100,
        min_clients_for_aggregation: int = 2,
        simulation_rounds: int = 3,
    ):
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.min_clients = min_clients_for_aggregation
        self.simulation_rounds = simulation_rounds
        
        self.server = None
        self.clients: List = []
        self.transport = None
        
        # Temporary directory for ONNX exports
        self.tmpdir = tempfile.mkdtemp()
    
    def setup(self) -> None:
        """Setup server and clients."""
        from federated.server import FederatedServer, ServerConfig
        from federated.transport import LocalTransport
        from federated.client import LocalTrainer, FederatedClient, FederatedClientConfig
        
        logger.info("=" * 60)
        logger.info("Setting up federated simulation")
        logger.info("  Clients: %d", self.num_clients)
        logger.info("  Samples per client: %d", self.samples_per_client)
        logger.info("  Min clients for aggregation: %d", self.min_clients)
        logger.info("=" * 60)
        
        # Create server
        server_config = ServerConfig(
            min_clients=self.min_clients,
            round_timeout_sec=30.0,
            model_class=DummySTGNN,
            model_kwargs={"in_channels": 5, "hidden_channels": 32},
            onnx_export_dir=self.tmpdir,
        )
        self.server = FederatedServer(server_config)
        
        # Create transport
        self.transport = LocalTransport(self.server)
        
        # Create clients
        for i in range(self.num_clients):
            device_id = f"edge-{i:03d}"
            
            # Mock edge client
            edge_client = MockEdgeClient(
                device_id=device_id,
                num_samples=self.samples_per_client,
            )
            
            # Trainer
            trainer = LocalTrainer(
                model_class=DummySTGNN,
                model_kwargs={"in_channels": 5, "hidden_channels": 32},
                learning_rate=0.001,
            )
            
            # Federated client config
            fed_config = FederatedClientConfig(
                training_interval_sec=1.0,  # Fast for simulation
                heartbeat_interval_sec=5.0,
                max_local_epochs=2,
                min_samples_for_training=16,
                learning_rate=0.001,
                batch_size=8,
            )
            
            # Create federated client
            fed_client = FederatedClient(
                edge_client=edge_client,
                transport=self.transport,
                trainer=trainer,
                config=fed_config,
            )
            
            self.clients.append(fed_client)
        
        logger.info("Setup complete: %d clients created", len(self.clients))
    
    def run_simulation(self) -> None:
        """Run the federated simulation."""
        logger.info("\n" + "=" * 60)
        logger.info("Starting federated simulation")
        logger.info("=" * 60)
        
        # Register all clients
        logger.info("\n--- Phase 1: Registration ---")
        for client in self.clients:
            client._edge_client.initialize()
            client._register()
        
        # Run training rounds
        for round_num in range(self.simulation_rounds):
            logger.info("\n--- Round %d/%d ---", round_num + 1, self.simulation_rounds)
            
            # Each client trains and submits
            for client in self.clients:
                logger.info("[%s] Training...", client.device_id)
                
                # Refill training buffer for next round
                if len(client._edge_client.training_buffer) < 32:
                    client._edge_client.replace_training_buffer(
                        SyntheticTrainingBuffer(num_samples=self.samples_per_client)
                    )
                
                # Run one training cycle
                client._training_cycle()
            
            # Log server state
            stats = self.server.get_stats()
            logger.info(
                "Server: model_version=%d, pending_updates=%d",
                stats["model_version"],
                stats["pending_updates"],
            )
            
            # Poll for aggregated model
            for client in self.clients:
                client._poll_for_model()
            
            # Check if aggregation happened
            if self.server.model_version > round_num:
                logger.info("✓ Aggregation complete: v%d", self.server.model_version)
        
        logger.info("\n" + "=" * 60)
        logger.info("Simulation complete!")
        logger.info("=" * 60)
        
        # Print final statistics
        self._print_stats()
    
    def _print_stats(self) -> None:
        """Print simulation statistics."""
        logger.info("\n--- Final Statistics ---")
        
        # Server stats
        server_stats = self.server.get_stats()
        logger.info("Server:")
        logger.info("  Model version: %d", server_stats["model_version"])
        logger.info("  Total devices: %d", server_stats["registry"]["total_devices"])
        
        # Client stats
        logger.info("Clients:")
        for client in self.clients:
            stats = client.get_stats()
            logger.info(
                "  [%s] version=%d, rounds=%d, samples=%d",
                stats["device_id"],
                stats["model_version"],
                stats["training_rounds"],
                stats["samples_trained"],
            )
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        for client in self.clients:
            try:
                client.stop()
            except Exception:
                pass
        
        if self.server is not None:
            self.server.shutdown()
        
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(self.tmpdir)
        except Exception:
            pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run federated STGNN learning simulation"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Number of federated clients (default: 3)",
    )
    parser.add_argument(
        "--samples-per-client",
        type=int,
        default=100,
        help="Training samples per client (default: 100)",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum clients for aggregation (default: 2)",
    )
    parser.add_argument(
        "--simulation-rounds",
        type=int,
        default=3,
        help="Number of federated rounds to simulate (default: 3)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run simulation
    sim = FederatedSimulation(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        min_clients_for_aggregation=args.min_clients,
        simulation_rounds=args.simulation_rounds,
    )
    
    try:
        sim.setup()
        sim.run_simulation()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        sim.cleanup()


if __name__ == "__main__":
    main()
