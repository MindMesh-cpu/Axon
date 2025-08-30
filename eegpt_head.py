
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class HeadConfig:
    d_in: int
    num_classes: int
    dropout: float = 0.1
    temporal: str = "none"   # "none" | "dwconv" | "attn"
    attn_heads: int = 4

class TemporalAdapter(nn.Module):
    """把 [B,T,D] 压到 [B,D]；[B,D] 直接透传"""
    def __init__(self, d_in: int, mode: str = "none", attn_heads: int = 4):
        super().__init__()
        self.mode = mode
        if mode == "dwconv":
            self.block = nn.Conv1d(d_in, d_in, kernel_size=3, padding=1, groups=d_in)
        elif mode == "attn":
            self.block = nn.MultiheadAttention(embed_dim=d_in, num_heads=attn_heads, batch_first=True)
        else:
            self.block = None

    def forward(self, z):  # z: [B,D] or [B,T,D]
        if z.dim() == 2:
            return z
        B, T, D = z.shape
        if self.mode == "dwconv":
            x = z.transpose(1, 2)      # [B,D,T]
            x = self.block(x)          # depthwise conv
            return x.mean(-1)          # GAP -> [B,D]
        elif self.mode == "attn":
            a, _ = self.block(z, z, z) # [B,T,D]
            return a.mean(1)           # [B,D]
        else:
            return z.mean(1)           # 默认: 时间平均

class MLPHead(nn.Module):
    def __init__(self, d_in: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, z):               # z: [B,D]
        return self.net(z)

class AxonHead(nn.Module):
    """EEGPT 兼容的分类头：可选时序适配 + 轻量 MLP"""
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.temporal = TemporalAdapter(cfg.d_in, cfg.temporal, cfg.attn_heads)
        self.head = MLPHead(cfg.d_in, cfg.num_classes, cfg.dropout)

    def forward(self, z):               # z: [B,D] or [B,T,D]
        if z.dim() == 3:
            z = self.temporal(z)
        return self.head(z)

def build_head(**kwargs) -> nn.Module:
    return AxonHead(HeadConfig(**kwargs))


