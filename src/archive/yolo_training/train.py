import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from preprocess.dataloader import CrowdGraphDataset, collate_fn
from models.stgnn import STGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(pred, target, inp):
    """Compute evaluation metrics with proper per-batch tracking."""
    true_xy = target.x[:, :2]
    
    # MSE per coordinate (anomaly score)
    mse = torch.mean((pred - true_xy) ** 2).item()
    
    # Per-coordinate RMSE for interpretability
    rmse_x = torch.sqrt(torch.mean((pred[:, 0] - true_xy[:, 0]) ** 2)).item()
    rmse_y = torch.sqrt(torch.mean((pred[:, 1] - true_xy[:, 1]) ** 2)).item()
    
    # Number of people (for context)
    num_people = true_xy.shape[0]
    
    # Average movement magnitude from input
    dxdy = inp.x[:, 2:4]
    avg_movement = torch.mean(torch.sqrt(torch.sum(dxdy ** 2, dim=1) + 1e-8)).item()
    
    # Max prediction error (worst case)
    max_error = torch.max(torch.sqrt(torch.sum((pred - true_xy) ** 2, dim=1))).item()
    
    return {
        "mse": mse,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "num_people": num_people,
        "avg_movement": avg_movement,
        "max_error": max_error
    }


def train_epoch(model, loader, optimizer, criterion, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        if not batch:
            continue

        for inp, target in batch:
            # Validate inputs
            if inp.x is None or target.x is None:
                continue
            if inp.x.shape[0] == 0 or target.x.shape[0] == 0:
                continue
            if inp.x.shape[0] != target.x.shape[0]:
                continue

            try:
                x_in = inp.x.unsqueeze(0).unsqueeze(0).to(device)
                edge_index = inp.edge_index.to(device)
                y_true = target.x[:, :2].to(device)

                optimizer.zero_grad()
                y_pred = model(x_in, edge_index).squeeze(0)

                loss = criterion(y_pred, y_true)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] NaN/Inf loss, skipping")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                
            except RuntimeError as e:
                print(f"[WARNING] Runtime error: {e}")
                continue

    return total_loss / max(1, n_batches)


def evaluate(model, loader, criterion, verbose=False):
    """Evaluate model with detailed per-sequence metrics."""
    model.eval()
    total_loss = 0
    all_metrics = []
    sequence_losses = []
    n_batches = 0

    with torch.no_grad():
        for seq_idx, batch in enumerate(loader):
            if not batch:
                continue

            seq_loss = 0
            seq_count = 0
            
            for inp, target in batch:
                if inp.x is None or target.x is None:
                    continue
                if inp.x.shape[0] == 0 or target.x.shape[0] == 0:
                    continue
                if inp.x.shape[0] != target.x.shape[0]:
                    continue

                try:
                    x_in = inp.x.unsqueeze(0).unsqueeze(0).to(device)
                    edge_index = inp.edge_index.to(device)
                    y_true = target.x[:, :2].to(device)

                    y_pred = model(x_in, edge_index).squeeze(0)
                    loss = criterion(y_pred, y_true)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    total_loss += loss.item()
                    seq_loss += loss.item()
                    n_batches += 1
                    seq_count += 1

                    metrics = compute_metrics(y_pred.cpu(), target, inp)
                    all_metrics.append(metrics)
                    
                except RuntimeError as e:
                    continue
            
            # Track per-sequence loss
            if seq_count > 0:
                avg_seq_loss = seq_loss / seq_count
                sequence_losses.append((seq_idx, avg_seq_loss))
                
                if verbose and avg_seq_loss > 0.1:  # Flag suspicious sequences
                    print(f"  [!] Seq {seq_idx}: High loss = {avg_seq_loss:.4f}")

    avg_loss = total_loss / max(1, n_batches)
    
    # Compute aggregate statistics
    if all_metrics:
        avg_metrics = {
            "mse": np.mean([m["mse"] for m in all_metrics]),
            "rmse_x": np.mean([m["rmse_x"] for m in all_metrics]),
            "rmse_y": np.mean([m["rmse_y"] for m in all_metrics]),
            "num_people": np.mean([m["num_people"] for m in all_metrics]),
            "avg_movement": np.mean([m["avg_movement"] for m in all_metrics]),
            "max_error": np.max([m["max_error"] for m in all_metrics])
        }
    else:
        avg_metrics = None
    
    return avg_loss, avg_metrics, sequence_losses


def main():
    root = "D:/stgnn_project/data/processed/txt_graphs"

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    
    if not os.path.exists(train_dir):
        print(f"[ERROR] Train dir not found: {train_dir}")
        return
    if not os.path.exists(val_dir):
        print(f"[WARNING] Val dir not found: {val_dir}")
        os.makedirs(val_dir, exist_ok=True)

    train_dataset = CrowdGraphDataset(root, split="train")
    val_dataset = CrowdGraphDataset(root, split="val")

    print(f"[INFO] Train sequences: {len(train_dataset)}")
    print(f"[INFO] Val sequences: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("[ERROR] Train dataset empty!")
        return
    
    has_validation = len(val_dataset) > 0

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        collate_fn=collate_fn, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = STGNN(
        in_channels=5, 
        hidden_channels=64,
        out_channels=2, 
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model parameters: {model.count_parameters():,}")
    params, size_mb = model.get_model_size()
    print(f"[INFO] Model size: {size_mb:.2f} MB")

    # Training settings
    epochs = 15  # Increased from 10
    best_val_loss = float('inf')
    patience = 5  # Increased from 3
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}")
        
        if has_validation:
            # Verbose output every 5 epochs
            verbose = (epoch % 5 == 0)
            val_loss, metrics, seq_losses = evaluate(model, val_loader, criterion, verbose=verbose)
            
            print(f"Val Loss: {val_loss:.4f}")

            if metrics:
                print(f"Val Metrics →")
                print(f"  MSE: {metrics['mse']:.4f}")
                print(f"  RMSE: X={metrics['rmse_x']:.3f}, Y={metrics['rmse_y']:.3f}")
                print(f"  People: {metrics['num_people']:.1f}")
                print(f"  Movement: {metrics['avg_movement']:.3f}")
                print(f"  Max Error: {metrics['max_error']:.3f}")
            
            # Show top-3 worst sequences
            if seq_losses and verbose:
                seq_losses_sorted = sorted(seq_losses, key=lambda x: x[1], reverse=True)[:3]
                print(f"\nTop-3 Worst Sequences:")
                for seq_idx, loss in seq_losses_sorted:
                    print(f"  Seq {seq_idx}: Loss={loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics
                }, "checkpoints/stgnn_best.pth")
                print(f"[SAVE] Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"[INFO] No improvement for {patience_counter} epoch(s)")
            
            if patience_counter >= patience:
                print(f"[STOP] Early stopping at epoch {epoch}")
                break

    # Save final
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "checkpoints/stgnn_final.pth")
    
    print(f"\n[OK] Final model: checkpoints/stgnn_final.pth")
    if has_validation:
        print(f"[OK] Best model: checkpoints/stgnn_best.pth (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
