# Federated Protocol
"""
Communication protocol for federated STGNN learning.

Provides message types for edge-server communication.
"""

from .messages import (
    Message,
    StateDict,
    RegisterDevice,
    SubmitUpdate,
    Heartbeat,
    RegisterAck,
    AggregatedModel,
    UpdateAck,
    create_register_device,
    create_submit_update,
    create_heartbeat,
)

__all__ = [
    "Message",
    "StateDict",
    "RegisterDevice",
    "SubmitUpdate",
    "Heartbeat",
    "RegisterAck",
    "AggregatedModel",
    "UpdateAck",
    "create_register_device",
    "create_submit_update",
    "create_heartbeat",
]
