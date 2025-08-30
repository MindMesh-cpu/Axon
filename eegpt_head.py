from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class HeadConfig:
    d_in: int                 # Input feature dimension
    num_classes: int          # Number of output classes (e.g., 3 or 4 for MI)
    dropout: float = 0.1      # Dropout rate for regularization
    temporal: str = "none"    # Temporal aggregation method: "none", "dwconv", "attn"
    attn_heads: int = 4       # Number of attention heads (for "attn" mode)

class TemporalAdapter(nn.Module):
    """Adapt the temporal dimension [B,T,D] to [B,D]; [B,D] is passed through unchanged."""
    def __init__(self, d_in: int, mode: str = "none", attn_heads: int = 4):
        super().__init__()
        self.mode = mode
        if mode == "dwconv":
            self.block = nn.Conv1d(d_in, d_in, kernel_size=3, padding=1, groups=d_in)  # Depthwise convolution
        elif mode == "attn":
            self.block = nn.MultiheadAttention(embed_dim=d_in, num_heads=attn_heads, batch_first=True)  # Attention mechanism
        else:
            self.block = None  # No transformation

    def forward(self, z):  # z: [B,D] or [B,T,D]
        if z.dim() == 2:
            return z  # If the input is already [B,D], return it directly
        B, T, D = z.shape
        if self.mode == "dwconv":
            x = z.transpose(1, 2)  # [B,D,T] -> [B,T,D] for Conv1d
            x = self.block(x)      # Apply depthwise convolution
            return x.mean(-1)      # Global Average Pooling -> [B,D]
        elif self.mode == "attn":
            a, _ = self.block(z, z, z)  # Apply attention mechanism
            return a.mean(1)             # Attention output -> [B,D]
        else:
            return z.mean(1)  # Default: Average across time dimension [B,T,D] -> [B,D]

class MLPHead(nn.Module):
    """A simple multi-layer perceptron (MLP) classifier head."""
    def __init__(self, d_in: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),               # Normalize input
            nn.Linear(d_in, 256),              # Fully connected layer
            nn.ReLU(),                         # Activation function
            nn.Dropout(dropout),               # Dropout for regularization
            nn.Linear(256, num_classes)        # Output layer
        )

    def forward(self, z):  # z: [B,D]
        return self.net(z)

class AxonHead(nn.Module):
    """A classification head compatible with EEGPT: Optional temporal adapter + lightweight MLP."""
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.temporal = TemporalAdapter(cfg.d_in, cfg.temporal, cfg.attn_heads)  # Temporal processing (optional)
        self.head = MLPHead(cfg.d_in, cfg.num_classes, cfg.dropout)  # Final classification head

    def forward(self, z):  # z: [B,D] or [B,T,D]
        if z.dim() == 3:  # If the input is [B,T,D], apply temporal adapter
            z = self.temporal(z)
        return self.head(z)  # Pass through the MLP head

def build_head(**kwargs) -> nn.Module:
    """Build and return the classification head."""
    return AxonHead(HeadConfig(**kwargs))
