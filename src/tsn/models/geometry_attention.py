"""Pre-normalized attention blocks used by the replacement geometry modules."""
from torch import nn


class GeometryAttentionBlock(nn.Module):
    def __init__(self, width, cross=False):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.cross_norm = nn.LayerNorm(width) if cross else None
        self.cross_attention = nn.MultiheadAttention(width, 4, batch_first=True) if cross else None
        self.key_norm = nn.LayerNorm(width) if cross else None
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))

    def forward(self, tokens, context=None):
        normalized = self.norm(tokens)
        tokens = tokens+self.attention(normalized, normalized, normalized, need_weights=False)[0]
        if self.cross_attention is not None:
            keys = self.key_norm(context)
            tokens = tokens+self.cross_attention(self.cross_norm(tokens), keys, keys, need_weights=False)[0]
        return tokens+self.ffn(self.ffn_norm(tokens))
