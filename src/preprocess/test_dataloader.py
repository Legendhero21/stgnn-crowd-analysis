import torch
from torch.utils.data import DataLoader
from dataloader import CrowdGraphDataset  # adjust import if needed

# Path to processed graphs
root = "D:/stgnn_project/data/processed/txt_graphs"

# Create dataset
train_dataset = CrowdGraphDataset(root, split="train")
print(f"Train sequences: {len(train_dataset)}")

# Grab one sequence
sequence_samples = train_dataset.get(0)
print(f"Sequence has {len(sequence_samples)} frame pairs")

# Inspect first pair
inp, target = sequence_samples[0]
print("Input frame:", inp.frame_id, "| Nodes:", inp.x.shape, "| Edges:", inp.edge_index.shape)
print("Target frame:", target.frame_id, "| Nodes:", target.x.shape, "| Edges:", target.edge_index.shape)

# Wrap with DataLoader if you want batching
loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
batch = next(iter(loader))
print("Batch length (list of pairs):", len(batch[0]))
