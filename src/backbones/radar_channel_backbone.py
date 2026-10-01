import torch
import torch.nn as nn
from .imu_channel_backbone import ConvBlock1d


class CoarsePointPool(nn.Module):
    """Coarse pooling of points within a frame: 1D conv over the point axis
    (with a residual/skip connection), then mask-aware avg+max pooling.
    Replaces the previous PointNet point encoder.
    """

    def __init__(self, in_feats, hidden=32, out_dim=32):
        super().__init__()
        self.conv1 = ConvBlock1d(in_feats, hidden, k=3, stride=1, dilation=1)
        self.conv2 = ConvBlock1d(hidden, hidden, k=3, stride=1, dilation=1)
        self.proj = nn.Linear(hidden * 2, out_dim)  # avg + max concat

    def forward(self, pts, mask):
        # pts: [B, T, N, F], mask: [B, T, N] bool
        B, T, N, F = pts.shape
        x = (pts * mask.unsqueeze(-1)).view(B * T, N, F).transpose(1, 2)  # [B*T, F, N]
        h = self.conv1(x)
        h = self.conv2(h) + h                                              # skip connection
        h = h.transpose(1, 2).view(B, T, N, -1)                             # [B,T,N,hidden]

        m = mask.unsqueeze(-1)
        count = m.sum(2).clamp(min=1)
        avg = (h * m).sum(2) / count
        mx = h.masked_fill(~m, float("-inf")).max(2).values
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)

        pooled = self.proj(torch.cat([avg, mx], dim=-1))                    # [B,T,out_dim]
        return torch.cat([pooled, count.log1p()], dim=-1)                    # [B,T,out_dim+1]


class TemporalTransformer(nn.Module):
    """Transformer over the per-frame pooled embeddings, with a skip
    connection around the whole stack."""

    def __init__(self, dim, num_layers=2, heads=4, ff=None, max_len=256):
        super().__init__()
        ff = ff or dim * 2
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        h = x + self.pos[:, :x.shape[1]]
        return self.enc(h) + x  # skip connection around the transformer stack


class RadarBackbone(nn.Module):
    """Input: x [B, T, N, F+1] -- packed point features + validity mask
    (last channel), matching the loader's single-tensor output.

    Pipeline: coarse conv pooling per frame -> per-frame embedding ->
    transformer over the temporal axis -> pool to num_tokens.

    Deliberately small: radar carries the least signal and is often
    missing/noisy, so it gets the smallest parameter budget.

    Output: [B, num_tokens, embed_dim] -- same interface as before.
    """

    def __init__(self, in_feats=6, point_hidden=32, point_out=32,
                 embed_dim=128, num_tokens=4, tf_layers=2, tf_heads=4,
                 max_len=256):
        super().__init__()
        self.point_pool = CoarsePointPool(in_feats, point_hidden, point_out)
        self.in_proj = nn.Linear(point_out + 1, embed_dim)
        self.transformer = TemporalTransformer(
            embed_dim, num_layers=tf_layers, heads=tf_heads, max_len=max_len)
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, x):
        pts = x[..., :-1]                              # [B, T, N, F]
        mask = x[..., -1] > 0.5                          # [B, T, N] bool
        frame_emb = self.point_pool(pts, mask)      # [B, T, point_out+1]
        h = self.in_proj(frame_emb)                       # [B, T, embed_dim]
        h = self.transformer(h)                             # [B, T, embed_dim]
        f = h.transpose(1, 2)                                 # [B, embed_dim, T]
        return self.pool(f).transpose(1, 2)                    # [B, num_tokens, embed_dim]


def build(**kwargs):
    return RadarBackbone(**kwargs)


if __name__ == "__main__":
    m = build()
    B, T, N, F = 2, 64, 16, 6
    pts = torch.randn(B, T, N, F)
    mask = (torch.rand(B, T, N) > 0.5).float()
    mask[:, :, 0] = 1.0
    x = torch.cat([pts, mask.unsqueeze(-1)], dim=-1)
    y = m(x)
    print("tokens", tuple(y.shape))