import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolutional Block with residual connections."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super(STGCNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Spatial graph convolution
        self.gcn = GCNConv(in_channels, out_channels)
        self.bn_spatial = BatchNorm(out_channels)
        
        # Temporal convolution
        self.tconv = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn_temporal = BatchNorm(out_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection projection (if dimensions differ)
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: [B, T, N, F] - Batch, Time, Nodes, Features
            edge_index: [2, E] - Edge connections
        Returns:
            x: [B, T, N, out_channels]
        """
        B, T, N, in_features = x.shape
        
        # Input validation
        if B != 1:
            raise ValueError(f"Currently only batch_size=1 is supported, got {B}")
        if N == 0:
            raise ValueError("Number of nodes cannot be zero")
        if T == 0:
            raise ValueError("Number of timesteps cannot be zero")
        
        # Ensure edge_index is correct type and device
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x.device)
        else:
            edge_index = edge_index.to(torch.long).to(x.device)
        
        # Store input for residual connection
        residual = x
        
        # === Spatial Processing (per timestep) ===
        out = []
        for t in range(T):
            xt = x[:, t, :, :].squeeze(0)  # [N, in_features]
            
            if torch.isnan(xt).any() or torch.isinf(xt).any():
                print(f"[WARNING] NaN/Inf detected in input at timestep {t}")
                xt = torch.nan_to_num(xt, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Graph convolution
            try:
                xt = self.gcn(xt, edge_index)  # [N, out_channels]
                xt = self.bn_spatial(xt)
                xt = F.relu(xt)
                xt = self.dropout(xt)
            except Exception as e:
                print(f"[ERROR] GCN failed at timestep {t}: {e}")
                raise
            
            xt = xt.unsqueeze(0).unsqueeze(1)  # [1, 1, N, out_channels]
            out.append(xt)
        
        x = torch.cat(out, dim=1)  # [1, T, N, out_channels]
        
        # === Temporal Processing ===
        # Reshape: [1, T, N, out_channels] -> [N, out_channels, T]
        x = x.squeeze(0).permute(1, 2, 0)  # [N, out_channels, T]
        
        # Temporal convolution
        x = self.tconv(x)  # [N, out_channels, T]
        
        # Reshape back: [N, out_channels, T] -> [1, T, N, out_channels]
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1, T, N, out_channels]
        
        # Batch norm and activation
        B, T, N, out_features = x.shape
        x_flat = x.view(B * T * N, out_features)
        x_flat = self.bn_temporal(x_flat)
        x = x_flat.view(B, T, N, out_features)
        x = F.relu(x)
        x = self.dropout(x)
        
        # === Residual Connection ===
        if self.residual_proj is not None:
            # Project residual to match output dimensions
            residual = residual.view(B * T * N, in_features)
            residual = self.residual_proj(residual)
            residual = residual.view(B, T, N, self.out_channels)
        
        x = x + residual  # Skip connection
        
        return x


class STGNN(nn.Module):
    """Spatial-Temporal Graph Neural Network for crowd trajectory prediction."""
    
    def __init__(
        self, 
        in_channels=5, 
        hidden_channels=64,
        out_channels=2, 
        num_layers=3,
        dropout=0.1,
        kernel_size=3
    ):
        """
        Args:
            in_channels: Input node features (x, y, dx, dy, density) = 5
            hidden_channels: Hidden feature dimensions
            out_channels: Output node features (predict x, y) = 2
            num_layers: Number of ST-GCN blocks
            dropout: Dropout probability
            kernel_size: Temporal convolution kernel size
        """
        super(STGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # ST-GCN blocks
        self.st_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.st_blocks.append(
                STGCNBlock(in_ch, hidden_channels, kernel_size, dropout)
            )
        
        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [1, T, N, F] - Input features
            edge_index: [2, E] - Graph edges
        Returns:
            out: [1, N, 2] - Predicted (x, y) coordinates
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, T, N, F], got shape {x.shape}")
        
        B, T, N, input_features = x.shape
        
        if B != 1:
            raise ValueError(f"Batch size must be 1, got {B}")
        if T == 0 or N == 0:
            raise ValueError(f"Invalid dimensions: T={T}, N={N}")
        if input_features != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input features, got {input_features}")
        
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[WARNING] NaN/Inf detected in model input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Pass through ST-GCN blocks
        for i, block in enumerate(self.st_blocks):
            try:
                x = block(x, edge_index)
            except Exception as e:
                print(f"[ERROR] Block {i} failed: {e}")
                raise
        
        # Use last timestep for prediction
        out = x[:, -1, :, :]  # [1, N, hidden_channels]
        
        # Output projection
        out = self.fc_out(out)  # [1, N, 2]
        
        return out
    
    def get_model_size(self):
        """Calculate model size in MB."""
        param_size = sum(p.numel() for p in self.parameters())
        param_size_mb = param_size * 4 / (1024 ** 2)  # 4 bytes per float32
        return param_size, param_size_mb
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test model
    model = STGNN(
        in_channels=5,
        hidden_channels=64,
        out_channels=2,
        num_layers=3,
        dropout=0.1
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    params, size_mb = model.get_model_size()
    print(f"Model size: {size_mb:.2f} MB")
    
    # Test forward pass
    B, T, N, Features = 1, 5, 10, 5
    x = torch.randn(B, T, N, Features)
    edge_index = torch.randint(0, N, (2, 30))
    
    try:
        output = model(x, edge_index)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test passed!")
    except Exception as e:
        print(f"Model test failed: {e}")
