import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv


FEDERATED_EDGE_STGNN_KWARGS = {
    "in_channels": 8,
    "hidden_channels": 64,
    "out_channels": 1,
    "num_layers": 3,
    "dropout": 0.1,
    "kernel_size": 3,
}


class STGCNBlock(nn.Module):
    """Spatial-temporal graph convolutional block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gcn = GCNConv(in_channels, out_channels)
        self.bn_spatial = BatchNorm(out_channels)

        self.tconv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.bn_temporal = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)

    def _masked_batch_norm(self, norm_layer, x, valid_mask):
        """Apply batch norm using only valid node/time positions."""
        if valid_mask is None:
            return norm_layer(x)

        valid_mask = valid_mask.reshape(-1).to(torch.bool)
        out = torch.zeros_like(x)

        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return out

        if self.training and valid_count == 1:
            out[valid_mask] = x[valid_mask]
            return out

        out[valid_mask] = norm_layer(x[valid_mask])
        return out

    def forward(self, x, edge_index, mask_seq=None):
        """
        Args:
            x: [B, T, N, F]
            edge_index: [2, E]
            mask_seq: Optional [B, T, N] node-validity mask.
        Returns:
            [B, T, N, out_channels]
        """
        B, T, N, in_features = x.shape

        if B != 1:
            raise ValueError(f"Currently only batch_size=1 is supported, got {B}")
        if N == 0:
            raise ValueError("Number of nodes cannot be zero")
        if T == 0:
            raise ValueError("Number of timesteps cannot be zero")

        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x.device)
        else:
            edge_index = edge_index.to(torch.long).to(x.device)

        if mask_seq is not None:
            if mask_seq.dim() != 3 or mask_seq.shape != (B, T, N):
                raise ValueError(
                    f"mask_seq must be [B, T, N]={((B, T, N))}, got {tuple(mask_seq.shape)}"
                )
            mask_seq = mask_seq.to(device=x.device, dtype=x.dtype)
            x = x * mask_seq.unsqueeze(-1)

        residual = x

        # --- Batched spatial GCN: all T timesteps in one call ---
        # [1, T, N, F] -> [T*N, F]
        x_flat = x.squeeze(0).reshape(T * N, in_features)

        if torch.isnan(x_flat).any() or torch.isinf(x_flat).any():
            x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=1e6, neginf=-1e6)

        # Block-diagonal edge_index: no cross-timestep messages
        if edge_index.shape[1] > 0:
            offsets = torch.arange(T, device=edge_index.device) * N
            edge_index_batched = torch.cat(
                [edge_index + off for off in offsets], dim=1
            )
        else:
            edge_index_batched = edge_index

        x_flat = self.gcn(x_flat, edge_index_batched)

        if mask_seq is not None:
            valid_mask = mask_seq.squeeze(0).reshape(T * N) > 0.5
            x_flat = self._masked_batch_norm(self.bn_spatial, x_flat, valid_mask)
        else:
            x_flat = self.bn_spatial(x_flat)

        x_flat = F.relu(x_flat)
        x_flat = self.dropout(x_flat)

        if mask_seq is not None:
            x_flat = x_flat * valid_mask.to(dtype=x_flat.dtype).unsqueeze(-1)

        # [T*N, out_ch] -> [1, T, N, out_ch]
        x = x_flat.reshape(T, N, -1).unsqueeze(0)

        if mask_seq is not None:
            x = x * mask_seq.unsqueeze(-1)

        x = x.squeeze(0).permute(1, 2, 0)  # [N, out_channels, T]
        x = self.tconv(x)
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1, T, N, out_channels]
        if mask_seq is not None:
            x = x * mask_seq.unsqueeze(-1)

        B, T, N, out_features = x.shape
        x_flat = x.reshape(B * T * N, out_features)
        if mask_seq is not None:
            valid_temporal = mask_seq.reshape(B * T * N) > 0.5
            x_flat = self._masked_batch_norm(
                self.bn_temporal,
                x_flat,
                valid_temporal,
            )
        else:
            x_flat = self.bn_temporal(x_flat)
        x = x_flat.reshape(B, T, N, out_features)
        x = F.relu(x)
        x = self.dropout(x)
        if mask_seq is not None:
            x = x * mask_seq.unsqueeze(-1)

        if self.residual_proj is not None:
            residual = residual.reshape(B * T * N, in_features)
            residual = self.residual_proj(residual)
            residual = residual.reshape(B, T, N, self.out_channels)

        x = x + residual
        if mask_seq is not None:
            x = x * mask_seq.unsqueeze(-1)
        return x


class STGNN(nn.Module):
    """Spatial-temporal graph neural network for crowd anomaly prediction."""

    def __init__(
        self,
        in_channels=5,
        hidden_channels=64,
        out_channels=1,
        num_layers=3,
        dropout=0.1,
        kernel_size=3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.st_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.st_blocks.append(
                STGCNBlock(in_ch, hidden_channels, kernel_size, dropout)
            )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, mask_seq=None):
        """
        Args:
            x: [B, T, N, F]
            edge_index: [2, E]
            mask_seq: Optional [B, T, N] node-validity mask.
        Returns:
            [B, out_channels] crowd-level score.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, T, N, F], got shape {x.shape}")

        B, T, N, input_features = x.shape
        if B != 1:
            raise ValueError(f"Batch size must be 1, got {B}")
        if T == 0 or N == 0:
            raise ValueError(f"Invalid dimensions: T={T}, N={N}")
        if input_features != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input features, got {input_features}"
            )

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        if mask_seq is not None:
            if mask_seq.dim() != 3 or mask_seq.shape != (B, T, N):
                raise ValueError(
                    f"mask_seq must be [B, T, N]={((B, T, N))}, got {tuple(mask_seq.shape)}"
                )
            mask_seq = mask_seq.to(device=x.device, dtype=x.dtype)
            x = x * mask_seq.unsqueeze(-1)

        for block in self.st_blocks:
            x = block(x, edge_index, mask_seq)

        node_repr = x[:, -1, :, :]  # [B, N, hidden_channels]
        node_pred = self.fc_out(node_repr)  # [B, N, out_channels]

        if mask_seq is not None:
            node_mask = mask_seq[:, -1, :].to(node_pred.device).unsqueeze(-1)
            node_pred = node_pred * node_mask
            valid_count = node_mask.sum(dim=1).clamp(min=1.0)
            out = (node_pred * node_mask).sum(dim=1) / valid_count
        else:
            out = node_pred.mean(dim=1)

        return out

    def get_model_size(self):
        """Calculate model size in MB."""
        param_size = sum(p.numel() for p in self.parameters())
        param_size_mb = param_size * 4 / (1024 ** 2)
        return param_size, param_size_mb

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = STGNN(
        in_channels=5,
        hidden_channels=64,
        out_channels=1,
        num_layers=3,
        dropout=0.1,
    )

    print(f"Model parameters: {model.count_parameters():,}")
    _, size_mb = model.get_model_size()
    print(f"Model size: {size_mb:.2f} MB")

    B, T, N, features = 1, 5, 10, 5
    x = torch.randn(B, T, N, features)
    edge_index = torch.randint(0, N, (2, 30))

    try:
        output = model(x, edge_index)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test passed")
    except Exception as exc:
        print(f"Model test failed: {exc}")
