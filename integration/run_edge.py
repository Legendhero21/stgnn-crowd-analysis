import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
from pathlib import Path

# Setup paths
_project_root = Path(__file__).parent.parent
_src_dir = _project_root / "src"

sys.path.insert(0, str(_src_dir))

# Imports
from federated.client import FederatedClient, FederatedClientConfig, LocalTrainer
from federated.edge.client import EdgeClient
from federated.edge.config import create_simulation_config
from federated.transport import LocalTransport
from federated.server import FederatedServer, ServerConfig
from models.stgnn import STGNN, FEDERATED_EDGE_STGNN_KWARGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    device_id = args.device_id
    video_path = args.video

    print(f"[{device_id}] Starting edge client...")

    # TEMP: create local server (will remove later)
    server_config = ServerConfig(
        min_clients=1,
        enable_timeout_watcher=False,
        model_class=STGNN,
        model_kwargs=dict(FEDERATED_EDGE_STGNN_KWARGS),
    )
    server = FederatedServer(server_config)

    transport = LocalTransport(server)

    edge_config = create_simulation_config(
        video_source=video_path,
        base_dir=str(_project_root),
        device_id=device_id,
    )

    edge_client = EdgeClient(edge_config)

    client_config = FederatedClientConfig(
        training_interval_sec=5.0,
        heartbeat_interval_sec=10.0,
        max_local_epochs=2,
        min_samples_for_training=16,
        learning_rate=0.001,
        batch_size=1,
    )

    trainer = LocalTrainer(
        model_class=STGNN,
        model_kwargs=dict(FEDERATED_EDGE_STGNN_KWARGS),
        learning_rate=client_config.learning_rate,
    )

    client = FederatedClient(
        edge_client=edge_client,
        transport=transport,
        trainer=trainer,
        config=client_config,
    )

    success = client.start(blocking=False, start_federated_loop=True)

    if not success:
        print("Failed to start client")
        return

    print(f"[{device_id}] Running...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        client.stop()


if __name__ == "__main__":
    main()