"""
Unit Tests for Federated Server Components
------------------------------------------
Tests for Phase 3: Federated Server Core

Tests verify:
- FedAvg math correctness
- Version increment logic
- Device registration & staleness
- Aggregation trigger conditions
- ONNX export after aggregation

Run with: pytest tests/test_federated_server.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

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
    
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=2):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        # x: [B, T, N, F] -> just take last timestep for simplicity
        x = x[:, -1, :, :]  # [B, N, F]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # [B, N, out_channels]


def create_dummy_state_dict(seed: int = 0) -> Dict[str, torch.Tensor]:
    """Create a dummy state_dict for testing."""
    torch.manual_seed(seed)
    model = DummySTGNN()
    return {k: v.clone() for k, v in model.state_dict().items()}


# ============================================================
# Protocol Message Tests
# ============================================================

class TestProtocolMessages:
    """Tests for protocol messages."""
    
    def test_register_device_serialization(self):
        """Test RegisterDevice pickle serialization."""
        from federated.protocol import RegisterDevice
        
        msg = RegisterDevice(
            device_id="test-device-001",
            device_type="laptop",
            current_model_version=1,
        )
        
        # Serialize and deserialize
        serialized = msg.serialize()
        deserialized = RegisterDevice.deserialize(serialized)
        
        assert deserialized.device_id == msg.device_id
        assert deserialized.device_type == msg.device_type
        assert deserialized.current_model_version == msg.current_model_version
    
    def test_submit_update_validation(self):
        """Test SubmitUpdate validation."""
        from federated.protocol import SubmitUpdate
        
        state_dict = create_dummy_state_dict()
        
        # Valid
        msg = SubmitUpdate(
            device_id="test",
            state_dict=state_dict,
            num_samples=100,
            base_version=1,
        )
        assert msg.device_id == "test"
        
        # Invalid: empty device_id
        with pytest.raises(ValueError):
            SubmitUpdate(device_id="", state_dict=state_dict, num_samples=100)
        
        # Invalid: None state_dict
        with pytest.raises(ValueError):
            SubmitUpdate(device_id="test", state_dict=None, num_samples=100)
    
    def test_heartbeat_creation(self):
        """Test Heartbeat message creation."""
        from federated.protocol import create_heartbeat
        
        msg = create_heartbeat(
            device_id="dev-001",
            model_version=5,
            sample_count=1000,
            is_training=True,
        )
        
        assert msg.device_id == "dev-001"
        assert msg.current_model_version == 5
        assert msg.sample_count == 1000
        assert msg.is_training is True


# ============================================================
# Aggregator (FedAvg) Tests
# ============================================================

class TestAggregator:
    """Tests for FedAvg aggregator."""
    
    def test_fedavg_weighted_average(self):
        """Test FedAvg computes correct weighted average."""
        from federated.server import Aggregator
        
        # Create two state dicts with known values
        state1 = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        state2 = {"weight": torch.tensor([4.0, 5.0, 6.0])}
        
        aggregator = Aggregator()
        
        # Device 1: 100 samples
        aggregator.add_update("dev1", state1, num_samples=100, base_version=0)
        
        # Device 2: 300 samples
        aggregator.add_update("dev2", state2, num_samples=300, base_version=0)
        
        result = aggregator.aggregate()
        
        assert result.success
        assert result.num_clients == 2
        assert result.total_samples == 400
        
        # Expected: (100/400 * state1) + (300/400 * state2)
        # = 0.25 * [1,2,3] + 0.75 * [4,5,6]
        # = [0.25, 0.5, 0.75] + [3, 3.75, 4.5]
        # = [3.25, 4.25, 5.25]
        expected = torch.tensor([3.25, 4.25, 5.25])
        
        assert torch.allclose(
            result.aggregated_state_dict["weight"],
            expected,
            atol=1e-5,
        )
    
    def test_fedavg_equal_weights(self):
        """Test FedAvg with equal sample counts."""
        from federated.server import Aggregator
        
        state1 = {"w": torch.tensor([0.0, 0.0])}
        state2 = {"w": torch.tensor([2.0, 4.0])}
        
        aggregator = Aggregator()
        aggregator.add_update("dev1", state1, num_samples=50, base_version=0)
        aggregator.add_update("dev2", state2, num_samples=50, base_version=0)
        
        result = aggregator.aggregate()
        
        # Equal weighting: average should be [1.0, 2.0]
        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(result.aggregated_state_dict["w"], expected)
    
    def test_reject_duplicate_device(self):
        """Test that duplicate device updates replace previous."""
        from federated.server import Aggregator
        
        state1 = {"w": torch.tensor([1.0])}
        state2 = {"w": torch.tensor([5.0])}
        
        aggregator = Aggregator()
        aggregator.add_update("dev1", state1, num_samples=100, base_version=0)
        aggregator.add_update("dev1", state2, num_samples=100, base_version=0)
        
        # Should only have 1 update (replaced)
        assert aggregator.update_count == 1
        
        result = aggregator.aggregate()
        
        # Should use the second value
        assert torch.allclose(result.aggregated_state_dict["w"], torch.tensor([5.0]))
    
    def test_reject_zero_samples(self):
        """Test that updates with 0 samples are rejected."""
        from federated.server import Aggregator
        
        state = {"w": torch.tensor([1.0])}
        
        aggregator = Aggregator()
        accepted = aggregator.add_update("dev1", state, num_samples=0, base_version=0)
        
        assert not accepted
        assert aggregator.update_count == 0
    
    def test_min_clients_check(self):
        """Test min_clients requirement."""
        from federated.server import Aggregator
        
        state = {"w": torch.tensor([1.0])}
        
        aggregator = Aggregator()
        aggregator.add_update("dev1", state, num_samples=100, base_version=0)
        
        # Require 2 clients but only 1 submitted
        result = aggregator.aggregate(min_clients=2)
        
        assert not result.success
        assert "Not enough clients" in result.error_message
    
    def test_verify_fedavg_math(self):
        """Test the FedAvg verification utility."""
        from federated.server.aggregator import verify_fedavg_math
        
        updates = [
            ({"w": torch.tensor([1.0, 2.0])}, 100),
            ({"w": torch.tensor([3.0, 4.0])}, 100),
        ]
        
        aggregated, is_correct = verify_fedavg_math(updates)
        
        assert is_correct
        # Equal weights: average of [1,2] and [3,4] = [2,3]
        assert torch.allclose(aggregated["w"], torch.tensor([2.0, 3.0]))


# ============================================================
# Device Registry Tests
# ============================================================

class TestDeviceRegistry:
    """Tests for device registry."""
    
    def test_register_device(self):
        """Test device registration."""
        from federated.server import DeviceRegistry
        
        registry = DeviceRegistry()
        
        device = registry.register("dev-001", "laptop", model_version=0)
        
        assert device.device_id == "dev-001"
        assert device.device_type == "laptop"
        assert registry.count == 1
    
    def test_staleness_detection(self):
        """Test device staleness marking."""
        from federated.server import DeviceRegistry, DeviceStatus
        
        # Very short timeout for testing
        registry = DeviceRegistry(stale_timeout_sec=0.1)
        
        registry.register("dev-001")
        
        # Wait for timeout
        time.sleep(0.15)
        
        stale = registry.mark_stale_devices()
        
        assert "dev-001" in stale
        assert registry.get("dev-001").status == DeviceStatus.STALE
    
    def test_heartbeat_refreshes_status(self):
        """Test that heartbeat makes stale device active."""
        from federated.server import DeviceRegistry, DeviceStatus
        
        registry = DeviceRegistry(stale_timeout_sec=0.1)
        registry.register("dev-001")
        
        time.sleep(0.15)
        registry.mark_stale_devices()
        
        assert registry.get("dev-001").status == DeviceStatus.STALE
        
        # Heartbeat should refresh
        registry.update_heartbeat("dev-001")
        
        assert registry.get("dev-001").status == DeviceStatus.ACTIVE
    
    def test_get_active_devices(self):
        """Test filtering active devices."""
        from federated.server import DeviceRegistry, DeviceStatus
        
        registry = DeviceRegistry(stale_timeout_sec=1000)  # Long timeout
        
        registry.register("dev-001")
        registry.register("dev-002")
        
        # Manually set one as stale
        registry.get("dev-002").status = DeviceStatus.STALE
        
        active = registry.get_active_devices()
        
        assert len(active) == 1
        assert active[0].device_id == "dev-001"


# ============================================================
# Model Manager Tests
# ============================================================

class TestModelManager:
    """Tests for model manager."""
    
    def test_version_increment(self):
        """Test version increments on update."""
        from federated.server import ModelManager
        
        manager = ModelManager(
            model_class=DummySTGNN,
            model_kwargs={"in_channels": 5, "hidden_channels": 32},
        )
        
        assert manager.version == 0
        
        new_state = create_dummy_state_dict(seed=123)
        manager.update_weights(new_state)
        
        assert manager.version == 1
        
        manager.update_weights(new_state)
        
        assert manager.version == 2
    
    def test_explicit_version(self):
        """Test explicit version setting."""
        from federated.server import ModelManager
        
        manager = ModelManager(model_class=DummySTGNN)
        
        new_state = create_dummy_state_dict()
        manager.update_weights(new_state, new_version=5)
        
        assert manager.version == 5
    
    def test_version_must_increase(self):
        """Test that version must increase."""
        from federated.server import ModelManager
        
        manager = ModelManager(model_class=DummySTGNN)
        
        new_state = create_dummy_state_dict()
        manager.update_weights(new_state)  # v1
        
        with pytest.raises(ValueError, match="must be > current"):
            manager.update_weights(new_state, new_version=0)
    
    def test_onnx_export(self):
        """Test ONNX export creates file."""
        from federated.server import ModelManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(
                model_class=DummySTGNN,
                onnx_export_dir=tmpdir,
            )
            
            onnx_path = manager.export_onnx()
            
            assert os.path.isfile(onnx_path)
            assert "v0" in onnx_path
            assert onnx_path.endswith(".onnx")
    
    def test_state_dict_validation(self):
        """Test state_dict key validation."""
        from federated.server import ModelManager
        
        manager = ModelManager(model_class=DummySTGNN)
        
        # Valid state dict
        valid = create_dummy_state_dict()
        assert manager.validate_state_dict(valid)
        
        # Invalid: extra key
        invalid = {**valid, "extra_key": torch.tensor([1.0])}
        assert not manager.validate_state_dict(invalid)
        
        # Invalid: missing key
        invalid = {k: v for k, v in valid.items() if "fc1" not in k}
        assert not manager.validate_state_dict(invalid)


# ============================================================
# Federated Server Tests
# ============================================================

class TestFederatedServer:
    """Tests for federated server orchestrator."""
    
    def _create_server(self, min_clients=2, timeout=60.0, stop_watcher=True):
        """
        Create a test server instance.
        
        Args:
            min_clients: Minimum clients for aggregation.
            timeout: Round timeout in seconds.
            stop_watcher: If True, stop the background watcher immediately.
                          Set to False for tests that need automatic timeout.
        
        Note:
            Call server.shutdown() in test cleanup if stop_watcher=False.
        """
        from federated.server import FederatedServer, ServerConfig
        
        config = ServerConfig(
            min_clients=min_clients,
            round_timeout_sec=timeout,
            model_class=DummySTGNN,
            model_kwargs={"in_channels": 5},
        )
        
        server = FederatedServer(config)
        
        # Stop watcher by default to avoid thread interference in most tests
        if stop_watcher:
            server._stop_timeout_watcher()
        
        return server
    
    def test_device_registration(self):
        """Test device registration flow."""
        from federated.protocol import RegisterDevice
        
        server = self._create_server()
        
        msg = RegisterDevice(
            device_id="edge-001",
            device_type="laptop",
            current_model_version=0,
        )
        
        ack = server.register_device(msg)
        
        assert ack.success
        assert ack.device_id == "edge-001"
        assert ack.current_global_version == 0
        assert server.registry.count == 1
    
    def test_aggregation_trigger_on_min_clients(self):
        """Test aggregation triggers when min_clients submit."""
        from federated.protocol import RegisterDevice, SubmitUpdate
        
        server = self._create_server(min_clients=2)
        state = create_dummy_state_dict()
        
        # Register devices
        for i in range(2):
            server.register_device(RegisterDevice(
                device_id=f"dev-{i}",
                device_type="laptop",
            ))
        
        # Submit from first device
        ack1 = server.submit_update(SubmitUpdate(
            device_id="dev-0",
            state_dict=state,
            num_samples=100,
            base_version=0,
        ))
        
        assert ack1.success
        assert server.model_version == 0  # Not aggregated yet
        
        # Submit from second device - should trigger aggregation
        ack2 = server.submit_update(SubmitUpdate(
            device_id="dev-1",
            state_dict=state,
            num_samples=100,
            base_version=0,
        ))
        
        assert ack2.success
        assert server.model_version == 1  # Aggregated!
    
    def test_aggregation_produces_onnx(self):
        """Test that aggregation produces ONNX file."""
        from federated.protocol import RegisterDevice, SubmitUpdate
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from federated.server import FederatedServer, ServerConfig
            
            config = ServerConfig(
                min_clients=1,
                model_class=DummySTGNN,
                onnx_export_dir=tmpdir,
            )
            server = FederatedServer(config)
            
            server.register_device(RegisterDevice(device_id="dev-0"))
            
            server.submit_update(SubmitUpdate(
                device_id="dev-0",
                state_dict=create_dummy_state_dict(),
                num_samples=50,
                base_version=0,
            ))
            
            # Check ONNX was exported
            stats = server.get_stats()
            onnx_path = stats["model_info"]["onnx_path"]
            
            assert onnx_path is not None
            assert os.path.isfile(onnx_path)
    
    def test_model_distribution(self):
        """Test that aggregated model is queued for distribution."""
        from federated.protocol import RegisterDevice, SubmitUpdate
        
        server = self._create_server(min_clients=1)
        state = create_dummy_state_dict()
        
        server.register_device(RegisterDevice(device_id="dev-0"))
        
        server.submit_update(SubmitUpdate(
            device_id="dev-0",
            state_dict=state,
            num_samples=100,
            base_version=0,
        ))
        
        # Model should be queued for distribution
        model = server.get_aggregated_model("dev-0")
        
        assert model is not None
        assert model.version == 1
        assert model.state_dict is not None
    
    def test_unregistered_device_rejected(self):
        """Test that updates from unregistered devices are rejected."""
        from federated.protocol import SubmitUpdate
        
        server = self._create_server()
        
        ack = server.submit_update(SubmitUpdate(
            device_id="unknown-device",
            state_dict=create_dummy_state_dict(),
            num_samples=100,
            base_version=0,
        ))
        
        assert not ack.success
        assert "not registered" in ack.error_message
    
    def test_round_timeout_manual_check(self):
        """Test round timeout triggers aggregation via manual check."""
        from federated.protocol import RegisterDevice, SubmitUpdate
        
        # Very short timeout, watcher stopped for deterministic test
        server = self._create_server(min_clients=10, timeout=0.1, stop_watcher=True)
        state = create_dummy_state_dict()
        
        server.register_device(RegisterDevice(device_id="dev-0"))
        
        # Submit one update (fewer than min_clients)
        server.submit_update(SubmitUpdate(
            device_id="dev-0",
            state_dict=state,
            num_samples=100,
            base_version=0,
        ))
        
        assert server.model_version == 0  # Not aggregated yet
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Manual check triggers aggregation
        triggered = server.check_round_timeout()
        
        assert triggered
        assert server.model_version == 1  # Now aggregated
    
    def test_automatic_timeout_trigger(self):
        """
        Test that timeout triggers aggregation AUTOMATICALLY via background watcher.
        
        This tests the production behavior where no external polling is needed.
        """
        from federated.protocol import RegisterDevice, SubmitUpdate
        
        # Very short timeout, watcher ENABLED
        server = self._create_server(min_clients=10, timeout=0.1, stop_watcher=False)
        state = create_dummy_state_dict()
        
        try:
            server.register_device(RegisterDevice(device_id="dev-0"))
            
            # Submit one update (fewer than min_clients)
            server.submit_update(SubmitUpdate(
                device_id="dev-0",
                state_dict=state,
                num_samples=100,
                base_version=0,
            ))
            
            assert server.model_version == 0  # Not aggregated yet
            
            # Wait for timeout + watcher interval
            # Watcher interval is min(1.0, 0.1/5) = 0.02s
            # So we wait timeout + some margin
            time.sleep(0.25)
            
            # Should have triggered automatically
            assert server.model_version == 1  # Aggregated automatically!
        finally:
            server.shutdown()
    
    def test_aggregation_callback(self):
        """Test aggregation callback is invoked."""
        from federated.protocol import RegisterDevice, SubmitUpdate
        
        server = self._create_server(min_clients=1)
        
        callback_called = []
        
        def on_aggregation(model):
            callback_called.append(model.version)
        
        server.set_aggregation_callback(on_aggregation)
        
        server.register_device(RegisterDevice(device_id="dev-0"))
        
        server.submit_update(SubmitUpdate(
            device_id="dev-0",
            state_dict=create_dummy_state_dict(),
            num_samples=50,
            base_version=0,
        ))
        
        assert callback_called == [1]
    
    def test_aggregation_callback_contract(self):
        """
        Test that callback is invoked ONLY AFTER:
        1. Aggregation completes
        2. Model version increments
        3. ONNX export succeeds
        4. Model is queued for distribution
        """
        from federated.protocol import RegisterDevice, SubmitUpdate
        import os
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from federated.server import FederatedServer, ServerConfig
            
            config = ServerConfig(
                min_clients=1,
                model_class=DummySTGNN,
                onnx_export_dir=tmpdir,
            )
            server = FederatedServer(config)
            server._stop_timeout_watcher()  # Stop for deterministic test
            
            callback_data = []
            
            def on_aggregation(model):
                # At callback time, verify all preconditions are met
                callback_data.append({
                    "callback_version": model.version,
                    "server_version": server.model_version,
                    "onnx_exists": os.path.isfile(model.onnx_path) if model.onnx_path else False,
                    "has_state_dict": model.state_dict is not None,
                    "distribution_queued": server.get_aggregated_model("dev-0") is not None,
                })
            
            server.set_aggregation_callback(on_aggregation)
            server.register_device(RegisterDevice(device_id="dev-0"))
            
            server.submit_update(SubmitUpdate(
                device_id="dev-0",
                state_dict=create_dummy_state_dict(),
                num_samples=50,
                base_version=0,
            ))
            
            # Verify callback was called and all conditions were met
            assert len(callback_data) == 1
            data = callback_data[0]
            assert data["callback_version"] == 1  # Correct version
            assert data["server_version"] == 1    # Server version incremented
            assert data["onnx_exists"]            # ONNX exported
            assert data["has_state_dict"]         # State dict present
            assert data["distribution_queued"]    # Distribution queued
    
    def test_server_shutdown(self):
        """Test server shutdown stops the timeout watcher."""
        # Create server with watcher enabled
        server = self._create_server(min_clients=2, timeout=60.0, stop_watcher=False)
        
        # Verify watcher is running
        assert server._timeout_watcher_thread is not None
        assert server._timeout_watcher_thread.is_alive()
        
        # Shutdown
        server.shutdown()
        
        # Verify watcher stopped
        assert server._timeout_watcher_thread is None


# ============================================================
# Integration Test
# ============================================================

class TestEndToEndFederated:
    """End-to-end integration test."""
    
    def test_full_federated_round(self):
        """Test complete federated round with multiple devices."""
        from federated.protocol import RegisterDevice, SubmitUpdate
        from federated.server import FederatedServer, ServerConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ServerConfig(
                min_clients=3,
                model_class=DummySTGNN,
                model_kwargs={"in_channels": 5},
                onnx_export_dir=tmpdir,
            )
            server = FederatedServer(config)
            server._stop_timeout_watcher()  # Stop for deterministic test
            
            try:
                # Register 3 devices
                for i in range(3):
                    ack = server.register_device(RegisterDevice(
                        device_id=f"device-{i}",
                        device_type="laptop",
                    ))
                    assert ack.success
                
                # Each device submits with different weights
                for i in range(3):
                    state = create_dummy_state_dict(seed=i * 100)
                    
                    ack = server.submit_update(SubmitUpdate(
                        device_id=f"device-{i}",
                        state_dict=state,
                        num_samples=100 + i * 50,  # Different sample counts
                        base_version=0,
                    ))
                    assert ack.success
                
                # Verify aggregation occurred
                assert server.model_version == 1
                
                # Verify ONNX exists
                stats = server.get_stats()
                assert os.path.isfile(stats["model_info"]["onnx_path"])
                
                # Verify all devices got the model queued
                for i in range(3):
                    model = server.get_aggregated_model(f"device-{i}")
                    assert model is not None
                    assert model.version == 1
                    assert model.participating_devices == 3
            finally:
                server.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
