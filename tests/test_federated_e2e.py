"""
End-to-End Tests for Federated Learning
----------------------------------------
Tests for Phase 4: Edge ↔ Server Integration

Tests verify:
- One full federated round completes
- EdgeClient ONNX model version updates after aggregation
- Multiple edges converge to same model version
- No deadlocks or race conditions

Run with: pytest tests/test_federated_e2e.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

# Add src to path
_src_dir = Path(__file__).parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


# ============================================================
# Test Fixtures
# ============================================================

class DummySTGNN(nn.Module):
    """Minimal STGNN-like model for testing."""
    
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=2, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = x[:, -1, :, :]  # [B, N, F]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MockTrainingBuffer:
    """Mock training buffer with synthetic data."""
    
    def __init__(self, num_samples: int = 50):
        self._samples = []
        for _ in range(num_samples):
            x_seq = torch.randn(1, 10, 20, 5)
            edge_index = torch.randint(0, 20, (2, 40))
            target = torch.randn(1, 20, 2)
            self._samples.append((x_seq, edge_index, target))
    
    def __len__(self):
        return len(self._samples)
    
    def iter_batches(self, batch_size: int):
        for i in range(0, len(self._samples), batch_size):
            batch = self._samples[i:i + batch_size]
            x_seqs = torch.cat([s[0] for s in batch], dim=0)
            edge_index = batch[0][1]
            targets = torch.cat([s[2] for s in batch], dim=0)
            yield x_seqs, edge_index, targets
    
    def clear(self):
        self._samples = []


class MockEdgeClient:
    """Mock EdgeClient for testing."""
    
    def __init__(self, device_id: str, num_samples: int = 50):
        self._device_id = device_id
        self._is_initialized = False
        self._training_buffer = MockTrainingBuffer(num_samples)
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
    
    @property
    def model_version(self) -> int:
        return self._model_version
    
    def initialize(self) -> bool:
        self._is_initialized = True
        return True
    
    def start(self, blocking: bool = False) -> None:
        pass
    
    def stop(self, timeout: float = 5.0) -> None:
        pass
    
    def update_onnx_model(self, new_model_path: str, new_version: int) -> None:
        self._model_version = new_version
    
    def replace_training_buffer(self, new_buffer) -> None:
        """Replace training buffer with new one."""
        self._training_buffer = new_buffer


def create_dummy_state_dict(seed: int = 0) -> Dict[str, torch.Tensor]:
    """Create a dummy state_dict for testing."""
    torch.manual_seed(seed)
    model = DummySTGNN()
    return {k: v.clone() for k, v in model.state_dict().items()}


# ============================================================
# LocalTransport Tests
# ============================================================

class TestLocalTransport:
    """Tests for LocalTransport adapter."""
    
    def test_register_device(self):
        """Test device registration via transport."""
        from federated.server import FederatedServer, ServerConfig
        from federated.transport import LocalTransport
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ServerConfig(
                min_clients=1,
                model_class=DummySTGNN,
                onnx_export_dir=tmpdir,
            )
            server = FederatedServer(config)
            server._stop_timeout_watcher()
            
            transport = LocalTransport(server)
            
            ack = transport.register_device(
                device_id="test-001",
                device_type="laptop",
                current_model_version=0,
            )
            
            assert ack.success
            assert ack.device_id == "test-001"
    
    def test_submit_and_poll(self):
        """Test update submission and model polling."""
        from federated.server import FederatedServer, ServerConfig
        from federated.transport import LocalTransport
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ServerConfig(
                min_clients=1,
                model_class=DummySTGNN,
                onnx_export_dir=tmpdir,
            )
            server = FederatedServer(config)
            server._stop_timeout_watcher()
            
            transport = LocalTransport(server)
            
            # Register
            transport.register_device("dev-001")
            
            # Submit update
            state_dict = create_dummy_state_dict()
            ack = transport.submit_update(
                device_id="dev-001",
                state_dict=state_dict,
                num_samples=100,
                base_version=0,
            )
            
            assert ack.success
            
            # Poll for model
            model = transport.poll_aggregated_model("dev-001")
            
            assert model is not None
            assert model.version == 1


# ============================================================
# LocalTrainer Tests
# ============================================================

class TestLocalTrainer:
    """Tests for LocalTrainer."""
    
    def test_train_with_synthetic_data(self):
        """Test training with synthetic data."""
        from federated.client import LocalTrainer
        
        trainer = LocalTrainer(
            model_class=DummySTGNN,
            model_kwargs={"in_channels": 5},
            learning_rate=0.001,
        )
        
        buffer = MockTrainingBuffer(num_samples=50)
        initial_weights = create_dummy_state_dict()
        
        result = trainer.train(
            training_buffer=buffer,
            initial_state_dict=initial_weights,
            max_epochs=2,
            batch_size=8,
            min_samples=16,
        )
        
        assert result.success
        assert result.samples_used > 0
        assert result.state_dict is not None
    
    def test_train_insufficient_samples(self):
        """Test training fails with insufficient samples."""
        from federated.client import LocalTrainer
        
        trainer = LocalTrainer(
            model_class=DummySTGNN,
            learning_rate=0.001,
        )
        
        buffer = MockTrainingBuffer(num_samples=5)  # Too few
        
        result = trainer.train(
            training_buffer=buffer,
            max_epochs=2,
            batch_size=8,
            min_samples=32,  # Requires more than we have
        )
        
        assert not result.success
        assert "Not enough samples" in result.error_message


# ============================================================
# Full Federated Round Tests
# ============================================================

class TestFederatedRound:
    """Tests for complete federated learning rounds."""
    
    def _create_server_and_transport(self, tmpdir: str, min_clients: int = 2):
        """Helper to create server and transport."""
        from federated.server import FederatedServer, ServerConfig
        from federated.transport import LocalTransport
        
        config = ServerConfig(
            min_clients=min_clients,
            round_timeout_sec=30.0,
            model_class=DummySTGNN,
            model_kwargs={"in_channels": 5},
            onnx_export_dir=tmpdir,
        )
        server = FederatedServer(config)
        server._stop_timeout_watcher()
        
        transport = LocalTransport(server)
        return server, transport
    
    def _create_federated_client(self, device_id: str, transport, num_samples: int = 50):
        """Helper to create a federated client."""
        from federated.client import LocalTrainer, FederatedClient, FederatedClientConfig
        
        edge_client = MockEdgeClient(device_id, num_samples)
        
        trainer = LocalTrainer(
            model_class=DummySTGNN,
            model_kwargs={"in_channels": 5},
            learning_rate=0.001,
        )
        
        config = FederatedClientConfig(
            training_interval_sec=0.1,
            heartbeat_interval_sec=1.0,
            max_local_epochs=2,
            min_samples_for_training=16,
            batch_size=8,
        )
        
        return FederatedClient(
            edge_client=edge_client,
            transport=transport,
            trainer=trainer,
            config=config,
        )
    
    def test_single_round_completes(self):
        """Test that a single federated round completes end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server, transport = self._create_server_and_transport(tmpdir, min_clients=2)
            
            try:
                # Create two clients
                client1 = self._create_federated_client("dev-001", transport)
                client2 = self._create_federated_client("dev-002", transport)
                
                # Initialize and register
                client1._edge_client.initialize()
                client2._edge_client.initialize()
                
                assert client1._register()
                assert client2._register()
                
                # Both run training cycle
                client1._training_cycle()
                client2._training_cycle()
                
                # Check aggregation happened
                assert server.model_version == 1
                
                # Check ONNX was exported
                stats = server.get_stats()
                assert stats["model_info"]["onnx_path"] is not None
                
            finally:
                server.shutdown()
    
    def test_model_version_sync_after_aggregation(self):
        """Test that clients sync to new model version after aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server, transport = self._create_server_and_transport(tmpdir, min_clients=2)
            
            try:
                client1 = self._create_federated_client("dev-001", transport)
                client2 = self._create_federated_client("dev-002", transport)
                
                # Initialize and register
                client1._edge_client.initialize()
                client2._edge_client.initialize()
                client1._register()
                client2._register()
                
                # Initial version is 0
                assert client1.model_version == 0
                assert client2.model_version == 0
                
                # Both train and submit
                client1._training_cycle()
                client2._training_cycle()
                
                # Poll for model
                client1._poll_for_model()
                client2._poll_for_model()
                
                # Both should have updated to v1
                assert client1.model_version == 1
                assert client2.model_version == 1
                
            finally:
                server.shutdown()
    
    def test_onnx_hotswap_triggered(self):
        """Test that ONNX hot-swap is triggered on clients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server, transport = self._create_server_and_transport(tmpdir, min_clients=1)
            
            try:
                client = self._create_federated_client("dev-001", transport)
                mock_edge = client._edge_client
                
                mock_edge.initialize()
                client._register()
                
                # Initial ONNX version is 0
                assert mock_edge.model_version == 0
                
                # Train and submit
                client._training_cycle()
                
                # Poll and apply
                client._poll_for_model()
                
                # ONNX should have been updated
                assert mock_edge.model_version == 1
                
            finally:
                server.shutdown()
    
    def test_multiple_edges_converge(self):
        """Test that multiple edges converge to the same model version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server, transport = self._create_server_and_transport(tmpdir, min_clients=3)
            
            try:
                # Create 3 clients
                clients = [
                    self._create_federated_client(f"dev-{i:03d}", transport)
                    for i in range(3)
                ]
                
                # Initialize and register all
                for c in clients:
                    c._edge_client.initialize()
                    c._register()
                
                # All train and submit
                for c in clients:
                    c._training_cycle()
                
                # Aggregation should have happened
                assert server.model_version == 1
                
                # All poll for model
                for c in clients:
                    c._poll_for_model()
                
                # All should have same version
                versions = [c.model_version for c in clients]
                assert all(v == 1 for v in versions)
                
            finally:
                server.shutdown()
    
    def test_no_deadlock_concurrent_operations(self):
        """Test that concurrent operations don't cause deadlocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server, transport = self._create_server_and_transport(tmpdir, min_clients=2)
            
            try:
                client1 = self._create_federated_client("dev-001", transport)
                client2 = self._create_federated_client("dev-002", transport)
                
                client1._edge_client.initialize()
                client2._edge_client.initialize()
                client1._register()
                client2._register()
                
                errors = []
                
                def run_client(client, num_cycles):
                    try:
                        for _ in range(num_cycles):
                            # Refill buffer
                            client._edge_client.replace_training_buffer(MockTrainingBuffer(50))
                            client._training_cycle()
                            client._poll_for_model()
                            time.sleep(0.01)
                    except Exception as e:
                        errors.append(str(e))
                
                # Run concurrently
                t1 = threading.Thread(target=run_client, args=(client1, 3))
                t2 = threading.Thread(target=run_client, args=(client2, 3))
                
                t1.start()
                t2.start()
                
                # Wait with timeout to detect deadlocks
                t1.join(timeout=10.0)
                t2.join(timeout=10.0)
                
                assert not t1.is_alive(), "Thread 1 deadlocked"
                assert not t2.is_alive(), "Thread 2 deadlocked"
                assert len(errors) == 0, f"Errors occurred: {errors}"
                
                # Model version should have increased
                assert server.model_version >= 1
                
            finally:
                server.shutdown()


# ============================================================
# Integration Test
# ============================================================

class TestFullIntegration:
    """Full integration test simulating real usage."""
    
    def test_three_rounds_three_clients(self):
        """Simulate 3 federated rounds with 3 clients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from federated.server import FederatedServer, ServerConfig
            from federated.transport import LocalTransport
            from federated.client import LocalTrainer, FederatedClient, FederatedClientConfig
            
            # Setup server
            config = ServerConfig(
                min_clients=3,
                model_class=DummySTGNN,
                model_kwargs={"in_channels": 5},
                onnx_export_dir=tmpdir,
            )
            server = FederatedServer(config)
            server._stop_timeout_watcher()
            
            transport = LocalTransport(server)
            
            # Create 3 clients
            clients = []
            for i in range(3):
                edge = MockEdgeClient(f"edge-{i}", num_samples=50)
                trainer = LocalTrainer(DummySTGNN, {"in_channels": 5})
                fed_config = FederatedClientConfig(
                    max_local_epochs=2,
                    min_samples_for_training=16,
                    batch_size=8,
                )
                client = FederatedClient(edge, transport, trainer, fed_config)
                clients.append(client)
            
            try:
                # Initialize and register
                for c in clients:
                    c._edge_client.initialize()
                    c._register()
                
                # Run 3 rounds
                for round_num in range(3):
                    # Refill buffers
                    for c in clients:
                        c._edge_client.replace_training_buffer(MockTrainingBuffer(50))
                    
                    # All train
                    for c in clients:
                        c._training_cycle()
                    
                    # All poll
                    for c in clients:
                        c._poll_for_model()
                
                # Verify 3 aggregations happened
                assert server.model_version == 3
                
                # All clients on v3
                for c in clients:
                    assert c.model_version == 3
                
            finally:
                server.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
