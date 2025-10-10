import torch
from torch_geometric.data import Data

# Allow PyTorch to load PyG Data objects
torch.serialization.add_safe_globals([Data])

graphs = torch.load(
    "D:/stgnn_project/data/processed/txt_graphs/train/00001.pt",
    weights_only=False
)

print(f"Frames in sequence: {len(graphs)}")
g = graphs[0]
print("Frame:", g.frame_id)
print("Nodes:", g.x.shape[0])
print("Node features shape:", g.x.shape)  # should be (N,2) or (N,5) if updated
print("Edges:", g.edge_index.shape)
print("Image:", g.image_path)
