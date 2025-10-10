import os
import torch
from torch_geometric.data import Dataset, Data
from torch.utils.data import DataLoader


class CrowdGraphDataset(Dataset):
    def __init__(self, root, split="train", transform=None, pre_transform=None):
        """
        Args:
            root (str): Base directory where processed .pt graph files are stored.
            split (str): "train", "val", or "test".
        """
        self.split_dir = os.path.join(root, split)
        
        # Check if directory exists
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Dataset directory not found: {self.split_dir}")
        
        self.files = [
            os.path.join(self.split_dir, f) 
            for f in os.listdir(self.split_dir) 
            if f.endswith(".pt")
        ]
        
        if not self.files:
            print(f"[WARNING] No .pt files found in {self.split_dir}")
        
        super().__init__(root, transform, pre_transform)

    def len(self):
        return len(self.files)

    def get(self, idx):
        """
        Load a sequence and create (input, target) pairs.
        With fixed-node graph_builder, all frames have matching node counts.
        """
        try:
            graphs = torch.load(self.files[idx], weights_only=False)
        except Exception as e:
            print(f"[ERROR] Failed to load {self.files[idx]}: {e}")
            return []
        
        if not graphs or len(graphs) < 2:
            return []
        
        # Build (input, target) pairs = (frame_t, frame_t+1)
        samples = []
        for i in range(len(graphs) - 1):
            inp = graphs[i]
            target = graphs[i + 1]
            
            # Sanity check (should always pass with fixed-node builder)
            if inp.x.shape[0] == target.x.shape[0]:
                samples.append((inp, target))
            else:
                # This should NEVER happen with the new graph_builder
                print(f"[WARNING] Unexpected node mismatch: frame {inp.frame_id} "
                      f"({inp.x.shape[0]} nodes) -> frame {target.frame_id} "
                      f"({target.x.shape[0]} nodes)")
        
        return samples


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Flattens a batch of sequences into a single list of (inp, target) pairs.
    """
    all_pairs = []
    for seq_samples in batch:
        if seq_samples:  # Skip empty sequences
            all_pairs.extend(seq_samples)
    
    if not all_pairs:
        return []
    
    return all_pairs


# Example usage and testing
if __name__ == "__main__":
    root = "D:/stgnn_project/data/processed/txt_graphs"
    
    # Test train dataset
    print("="*60)
    print("Testing Train Dataset")
    print("="*60)
    
    try:
        train_dataset = CrowdGraphDataset(root, split="train")
        print(f"[OK] Train sequences: {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            # Test loading first sequence
            first_seq = train_dataset.get(0)
            print(f"[OK] First sequence has {len(first_seq)} frame pairs")
            
            if first_seq:
                inp, target = first_seq[0]
                print(f"[OK] Sample pair: {inp.x.shape} -> {target.x.shape}")
            
            # Test dataloader
            train_loader = DataLoader(
                train_dataset, 
                batch_size=2, 
                collate_fn=collate_fn, 
                shuffle=True
            )
            
            batch = next(iter(train_loader))
            print(f"[OK] Batch size (pairs): {len(batch)}")
            
            if batch:
                inp, target = batch[0]
                print(f"[OK] Input frame {inp.frame_id}: {inp.x.shape}")
                print(f"[OK] Target frame {target.frame_id}: {target.x.shape}")
                print(f"[OK] Edge index: {inp.edge_index.shape}")
                
                # Verify node counts match
                assert inp.x.shape[0] == target.x.shape[0], "Node count mismatch!"
                print("[✓] Node counts match!")
    except Exception as e:
        print(f"[ERROR] Train dataset test failed: {e}")
    
    # Test val dataset
    print("\n" + "="*60)
    print("Testing Val Dataset")
    print("="*60)
    
    try:
        val_dataset = CrowdGraphDataset(root, split="val")
        print(f"[OK] Val sequences: {len(val_dataset)}")
        
        if len(val_dataset) > 0:
            first_seq = val_dataset.get(0)
            print(f"[OK] First sequence has {len(first_seq)} frame pairs")
    except Exception as e:
        print(f"[ERROR] Val dataset test failed: {e}")
