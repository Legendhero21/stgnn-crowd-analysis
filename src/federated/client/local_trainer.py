"""
Local Trainer
-------------
PyTorch training loop for edge devices in federated learning.

Responsibilities:
- Consume samples from TrainingBuffer
- Train a PyTorch STGNN model locally
- Return updated weights and training statistics

The trainer does NOT:
- Perform inference (that's handled by ONNX)
- Communicate with the server (that's FederatedClient's job)
- Modify the TrainingBuffer (samples are consumed via iteration)

Usage:
    trainer = LocalTrainer(
        model_class=STGNN,
        model_kwargs={"in_channels": 5, "hidden_channels": 64},
        learning_rate=0.001,
    )
    
    result = trainer.train(
        training_buffer=buffer,
        initial_state_dict=server_weights,
        max_epochs=5,
        batch_size=16,
    )
    
    # result.state_dict contains updated weights
    # result.samples_used contains number of samples trained on
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

logger = logging.getLogger(__name__)


# Type alias for state dict
StateDict = Dict[str, Tensor]


@dataclass
class TrainingResult:
    """
    Result of a local training session.
    
    Attributes:
        state_dict: Updated model weights after training.
        samples_used: Total number of samples used for training.
        epochs_completed: Number of epochs completed.
        final_loss: Loss value at end of training.
        training_time_sec: Time spent training in seconds.
        success: Whether training completed successfully.
        error_message: Error message if training failed.
    """
    state_dict: Optional[StateDict] = None
    samples_used: int = 0
    epochs_completed: int = 0
    final_loss: float = float("inf")
    training_time_sec: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class TrainingConfig:
    """
    Configuration for local training.
    
    Attributes:
        max_epochs: Maximum training epochs per round.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        min_samples: Minimum samples required to train.
        device: PyTorch device (cpu or cuda).
        loss_fn_name: Name of loss function (mse, l1, smooth_l1).
    """
    max_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    min_samples: int = 32
    device: str = "cpu"
    loss_fn_name: str = "mse"


class LocalTrainer:
    """
    Local PyTorch trainer for federated learning.
    
    Trains a PyTorch model using samples from TrainingBuffer.
    The trainer creates a fresh model instance for each training
    session, initialized with weights from the server.
    
    Thread Safety:
        NOT thread-safe. Each FederatedClient should have its own trainer.
    
    Attributes:
        model_class: PyTorch model class to instantiate.
        model_kwargs: Arguments for model instantiation.
        config: Training configuration.
    """
    
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.001,
        device: str = "cpu",
        loss_fn: str = "mse",
    ):
        """
        Initialize local trainer.
        
        Args:
            model_class: PyTorch model class (e.g., STGNN).
            model_kwargs: Arguments passed to model constructor.
            learning_rate: Learning rate for optimizer.
            device: PyTorch device ("cpu" or "cuda").
            loss_fn: Loss function name ("mse", "l1", "smooth_l1").
        """
        self._model_class = model_class
        self._model_kwargs = model_kwargs or {}
        self._learning_rate = learning_rate
        self._device = torch.device(device)
        self._loss_fn_name = loss_fn
        
        # Create loss function
        self._loss_fn = self._create_loss_fn(loss_fn)
        
        logger.info(
            "LocalTrainer initialized: model=%s, lr=%.4f, device=%s",
            model_class.__name__,
            learning_rate,
            device,
        )
    
    def _create_loss_fn(self, name: str) -> nn.Module:
        """Create loss function by name."""
        loss_map = {
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "smooth_l1": nn.SmoothL1Loss,
            "huber": nn.HuberLoss,
        }
        
        if name.lower() not in loss_map:
            raise ValueError(f"Unknown loss function: {name}")
        
        return loss_map[name.lower()]()
    
    def _create_model(self) -> nn.Module:
        """Create a fresh model instance."""
        model = self._model_class(**self._model_kwargs)
        model.to(self._device)
        return model
    
    def _prepare_batch(self, batch):
        """
        Convert a batch from iter_batches into (x_seq, edge_index, target) tensors.
        
        Supports two formats:
        - Real TrainingBuffer: yields List[TrainingSample] with numpy arrays
        - Mock buffer (tests): yields (x_seq, edge_index, target) torch tuples
        """
        if isinstance(batch, (list,)) and len(batch) > 0 and hasattr(batch[0], 'x_seq'):
            # Real TrainingBuffer: List[TrainingSample]
            x_seqs = torch.from_numpy(
                np.concatenate([s.x_seq for s in batch], axis=0)
            ).float()
            edge_index = torch.from_numpy(batch[0].edge_index).long()
            targets = torch.from_numpy(
                np.stack([s.target for s in batch], axis=0)
            ).float()
            return x_seqs, edge_index, targets
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            # Mock/test buffer: (x_seq_tensor, edge_index_tensor, target_tensor)
            return batch[0], batch[1], batch[2]
        else:
            raise TypeError(f"Unexpected batch format: {type(batch)}")
    
    def train(
        self,
        training_buffer: Any,
        initial_state_dict: Optional[StateDict] = None,
        max_epochs: int = 5,
        batch_size: int = 16,
        min_samples: int = 32,
    ) -> TrainingResult:
        """
        Train model using samples from buffer.
        
        Args:
            training_buffer: TrainingBuffer with collected samples.
            initial_state_dict: Weights to initialize model with.
                               If None, use random initialization.
            max_epochs: Maximum number of training epochs.
            batch_size: Batch size for each training step.
            min_samples: Minimum samples required to train.
        
        Returns:
            TrainingResult with updated weights and statistics.
        """
        start_time = time.time()
        
        # Check buffer has enough samples
        sample_count = len(training_buffer)
        if sample_count < min_samples:
            return TrainingResult(
                success=False,
                error_message=f"Not enough samples: {sample_count} < {min_samples}",
            )
        
        try:
            # Create model and load initial weights
            model = self._create_model()
            
            if initial_state_dict is not None:
                model.load_state_dict(initial_state_dict)
                logger.debug("Loaded initial weights from server")
            
            # Set to training mode
            model.train()
            
            # Create optimizer
            optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)
            
            # Training loop
            total_samples = 0
            final_loss = float("inf")
            
            for epoch in range(max_epochs):
                epoch_loss = 0.0
                epoch_samples = 0
                
                # Iterate over batches
                for batch in training_buffer.iter_batches(batch_size):
                    x_seq, edge_index, target = self._prepare_batch(batch)
                    
                    # Move to device
                    x_seq = x_seq.to(self._device)
                    edge_index = edge_index.to(self._device)
                    target = target.to(self._device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    prediction = model(x_seq, edge_index)
                    
                    # Compute loss
                    loss = self._loss_fn(prediction, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Accumulate statistics
                    batch_samples = x_seq.size(0) *  x_seq.size(2)
                    epoch_loss += loss.item() * batch_samples
                    epoch_samples += batch_samples
                
                if epoch_samples > 0:
                    epoch_loss /= epoch_samples
                    final_loss = epoch_loss
                    total_samples += epoch_samples
                    
                    logger.debug(
                        "Epoch %d/%d: loss=%.4f, samples=%d",
                        epoch + 1, max_epochs, epoch_loss, epoch_samples,
                    )
            
            # Extract updated state dict
            updated_state_dict = {
                k: v.clone().cpu()
                for k, v in model.state_dict().items()
            }
            
            training_time = time.time() - start_time
            
            logger.info(
                "Training complete: %d epochs, %d samples, loss=%.4f, time=%.1fs",
                max_epochs, total_samples, final_loss, training_time,
            )
            
            return TrainingResult(
                state_dict=updated_state_dict,
                samples_used=total_samples,
                epochs_completed=max_epochs,
                final_loss=final_loss,
                training_time_sec=training_time,
                success=True,
            )
            
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            return TrainingResult(
                success=False,
                error_message=str(exc),
                training_time_sec=time.time() - start_time,
            )
    
    def validate(
        self,
        state_dict: StateDict,
        validation_data: List[Tuple[Tensor, Tensor, Tensor]],
    ) -> float:
        """
        Validate model on held-out data.
        
        Args:
            state_dict: Model weights to validate.
            validation_data: List of (x_seq, edge_index, target) tuples.
        
        Returns:
            Average validation loss.
        """
        model = self._create_model()
        model.load_state_dict(state_dict)
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for x_seq, edge_index, target in validation_data:
                x_seq = x_seq.to(self._device)
                edge_index = edge_index.to(self._device)
                target = target.to(self._device)
                
                prediction = model(x_seq, edge_index)
                loss = self._loss_fn(prediction, target)
                
                batch_size = x_seq.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        return total_loss / total_samples if total_samples > 0 else float("inf")
