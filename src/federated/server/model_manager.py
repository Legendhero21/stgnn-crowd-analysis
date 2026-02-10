"""
Model Manager
-------------
Manages the global STGNN model for federated learning.

Responsibilities:
- Hold the global PyTorch STGNN model
- Track model version (monotonically increasing)
- Load initial weights from disk
- Apply aggregated weights
- Export PyTorch → ONNX for edge distribution
- Never performs inference (inference is edge-only via ONNX)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn

# Type alias
StateDict = Dict[str, Any]


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about the current global model."""
    version: int
    param_count: int
    param_keys: Set[str]
    last_updated: float
    onnx_path: Optional[str] = None


class ModelManager:
    """
    Manages the global PyTorch STGNN model.
    
    Thread-safe operations for:
    - Version tracking
    - Weight updates
    - ONNX export
    
    Usage:
        manager = ModelManager(
            model_class=STGNNModel,
            model_kwargs={"in_channels": 5, "hidden_channels": 64},
            initial_weights_path="models/stgnn_init.pt",
        )
        
        # Get current weights for distribution
        state_dict = manager.get_state_dict()
        
        # Apply aggregated weights
        manager.update_weights(aggregated_state_dict, new_version=2)
        
        # Export to ONNX for edges
        onnx_path = manager.export_onnx("outputs/models")
    """
    
    def __init__(
        self,
        model_class: type,
        model_kwargs: Optional[Dict[str, Any]] = None,
        initial_weights_path: Optional[str] = None,
        onnx_export_dir: str = "outputs/federated/models",
        device: str = "cpu",
    ):
        """
        Initialize model manager.
        
        Args:
            model_class: PyTorch model class (e.g., STGNN).
            model_kwargs: Arguments for model instantiation.
            initial_weights_path: Optional path to initial weights.
            onnx_export_dir: Directory for ONNX exports.
            device: Device for model ("cpu" or "cuda").
        """
        self._model_class = model_class
        self._model_kwargs = model_kwargs or {}
        self._onnx_export_dir = onnx_export_dir
        self._device = device
        
        # Create model
        self._model = self._create_model()
        
        # Load initial weights if provided
        if initial_weights_path and os.path.isfile(initial_weights_path):
            self._load_weights(initial_weights_path)
            logger.info("Loaded initial weights from: %s", initial_weights_path)
        
        # Version tracking (starts at 0)
        self._version = 0
        self._last_updated = time.time()
        self._latest_onnx_path: Optional[str] = None
        
        # Cache param keys for validation
        self._param_keys = set(self._model.state_dict().keys())
        
        logger.info(
            "ModelManager initialized: %d params, version=%d",
            sum(p.numel() for p in self._model.parameters()),
            self._version,
        )
    
    def _create_model(self) -> nn.Module:
        """Create a new model instance."""
        model = self._model_class(**self._model_kwargs)
        model.to(self._device)
        model.eval()  # Server model is never trained, only aggregated
        return model
    
    def _load_weights(self, path: str) -> None:
        """Load weights from file."""
        state_dict = torch.load(path, map_location=self._device)
        
        # Handle potential wrapper keys
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        self._model.load_state_dict(state_dict)
    
    @property
    def version(self) -> int:
        """Current model version."""
        return self._version
    
    @property
    def param_keys(self) -> Set[str]:
        """Set of parameter keys in the model."""
        return self._param_keys.copy()
    
    @property
    def latest_onnx_path(self) -> Optional[str]:
        """Path to most recently exported ONNX model."""
        return self._latest_onnx_path
    
    def get_info(self) -> ModelInfo:
        """Get current model information."""
        return ModelInfo(
            version=self._version,
            param_count=sum(p.numel() for p in self._model.parameters()),
            param_keys=self._param_keys.copy(),
            last_updated=self._last_updated,
            onnx_path=self._latest_onnx_path,
        )
    
    def get_state_dict(self) -> StateDict:
        """
        Get a copy of the current model state_dict.
        
        Returns:
            Deep copy of model weights.
        """
        return {k: v.clone().cpu() for k, v in self._model.state_dict().items()}
    
    def validate_state_dict(self, state_dict: StateDict) -> bool:
        """
        Validate that a state_dict is compatible.
        
        Args:
            state_dict: Weights to validate.
        
        Returns:
            True if compatible with model architecture.
        """
        if state_dict is None:
            return False
        
        incoming_keys = set(state_dict.keys())
        
        if incoming_keys != self._param_keys:
            missing = self._param_keys - incoming_keys
            extra = incoming_keys - self._param_keys
            logger.warning(
                "State dict key mismatch - missing: %s, extra: %s",
                missing, extra,
            )
            return False
        
        return True
    
    def update_weights(
        self,
        state_dict: StateDict,
        new_version: Optional[int] = None,
    ) -> int:
        """
        Update model weights.
        
        Args:
            state_dict: New model weights.
            new_version: Explicit new version (if None, increments by 1).
        
        Returns:
            New version number.
        
        Raises:
            ValueError: If state_dict is invalid.
        """
        if not self.validate_state_dict(state_dict):
            raise ValueError("Invalid state_dict: key mismatch")
        
        # Move tensors to device
        device_state = {
            k: v.to(self._device) for k, v in state_dict.items()
        }
        
        self._model.load_state_dict(device_state)
        
        # Update version
        if new_version is not None:
            if new_version <= self._version:
                raise ValueError(
                    f"New version ({new_version}) must be > current ({self._version})"
                )
            self._version = new_version
        else:
            self._version += 1
        
        self._last_updated = time.time()
        
        logger.info("Model updated to version %d", self._version)
        
        return self._version
    
    def export_onnx(
        self,
        output_dir: Optional[str] = None,
        temporal_window: int = 5,
        num_nodes: int = 50,
        num_features: int = 5,
        num_edges: int = 100,
    ) -> str:
        """
        Export current model to ONNX format.
        
        Args:
            output_dir: Output directory (uses default if None).
            temporal_window: T dimension for dummy input.
            num_nodes: N dimension for dummy input.
            num_features: F dimension for dummy input.
            num_edges: E dimension for dummy edge_index.
        
        Returns:
            Path to exported ONNX file.
        """
        output_dir = output_dir or self._onnx_export_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create versioned filename
        onnx_filename = f"stgnn_global_v{self._version}.onnx"
        onnx_path = os.path.join(output_dir, onnx_filename)
        
        # Create dummy inputs matching STGNN signature
        dummy_x = torch.randn(
            1, temporal_window, num_nodes, num_features,
            device=self._device, dtype=torch.float32,
        )
        dummy_edge_index = torch.randint(
            0, num_nodes, (2, num_edges),
            device=self._device, dtype=torch.int64,
        )
        
        # Export to temporary file first, then move atomically
        temp_path = onnx_path + ".tmp"
        
        try:
            self._model.eval()
            
            torch.onnx.export(
                self._model,
                (dummy_x, dummy_edge_index),
                temp_path,
                input_names=["x", "edge_index"],
                output_names=["predictions"],
                dynamic_axes={
                    "x": {0: "batch", 2: "nodes"},
                    "edge_index": {1: "edges"},
                    "predictions": {0: "batch", 1: "nodes"},
                },
                opset_version=14,
                do_constant_folding=True,
            )
            
            # Atomic move
            shutil.move(temp_path, onnx_path)
            
            # Validate exported ONNX with dummy inference
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(onnx_path)
                
                # Run dummy inference
                dummy_x_np = dummy_x.cpu().numpy()
                dummy_edge_np = dummy_edge_index.cpu().numpy()
                
                _ = sess.run(
                    None,
                    {"x": dummy_x_np, "edge_index": dummy_edge_np},
                )
                
                logger.debug("ONNX validation passed")
                
            except Exception as validation_exc:
                # Validation failed - remove the exported file
                logger.error("ONNX validation failed: %s", validation_exc)
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
                raise RuntimeError(f"ONNX validation failed: {validation_exc}")
            
            self._latest_onnx_path = onnx_path
            
            logger.info("Exported ONNX model: %s", onnx_path)
            
            return onnx_path
            
        except Exception as exc:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error("ONNX export failed: %s", exc)
            raise
    
    def reset_to_initial(self, weights_path: str) -> None:
        """
        Reset model to initial weights.
        
        Args:
            weights_path: Path to initial weights.
        """
        self._load_weights(weights_path)
        self._version = 0
        self._last_updated = time.time()
        logger.info("Model reset to initial weights")
