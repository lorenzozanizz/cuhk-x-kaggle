import math

import torch
import torch.nn as nn


COCO17_EDGES = [
    (0, 1), (0, 4), (4, 5), (5, 6), (1, 2), (2, 3), (0, 7), (7, 8),
    (8,9 ), (9, 10), (8, 11), (8, 14), (11, 12), (12, 13),
    (14, 15), (15, 16),
]


def build_adjacency(num_joints, edges):
    A = torch.eye(num_joints)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(-1, keepdim=True).clamp(min=1)
    return A / deg  # row-normalized, GCN-style


def sinusoidal_position_encoding(t, d_model, device, dtype=torch.float32):
    position = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype)
                          * (-math.log(10000.0) / d_model))
    pe = torch.zeros(t, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SpatialGCNBlock(nn.Module):
    """One hop of message passing over the COCO-17 skeleton graph, applied
    independently per frame, with a residual connection so the block refines
    features instead of fully overwriting them -- without this, stacking
    blocks can wash out per-joint identity before the temporal Transformer
    sees it, which is the likely cause of the earlier regression."""

    def __init__(self, channels, num_joints, edges, dropout=0.1):
        super().__init__()
        self.register_buffer("A", build_adjacency(num_joints, edges))
        self.lin = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T, J, C]
        agg = torch.einsum("ij,btjc->btic", self.A.to(x.dtype), x)
        out = self.drop(torch.relu(self.lin(agg)))
        return self.norm(x + out)  # residual


class SkeletonTransformerBackbone(nn.Module):
    """Input [B, T, J, 4] (xyz + depth-presence flag) plus optional [B, T]
    padding mask. Pipeline: per-joint embed -> (optional) masked-reconstruction
    token substitution -> 1-2 spatial GCN hops over the skeleton graph ->
    flatten+project to d_model -> temporal Transformer -> Perceiver-style
    resampler to a fixed [num_tokens, embed_dim] output.

    Two entry points:
      - forward(x, mask): classification path, returns pooled [B, num_tokens, embed_dim].
      - forward_encoder(x, mask, joint_frame_mask): pretraining path, returns
        the temporal encoder's per-frame hidden state [B, T, d_model] *before*
        the resampler, for a reconstruction head to decode. joint_frame_mask
        marks which (t, j) entries were replaced with a learnable mask token,
        i.e. the masked-autoencoding objective (hide chunks of input, train the
        network to infer them from context -- forces it to learn real
        structure about how joints move together instead of just memorizing).
    """

    def __init__(self, num_joints=17, in_ch=4, joint_ch=32, d_model=96,
                 n_layers=8, n_heads=6, ff_mult=2, dropout=0.25,
                 embed_dim=160, num_tokens=4, n_spatial_blocks=2,
                 edges=COCO17_EDGES):
        super().__init__()
        self.num_joints = num_joints
        self.joint_embed = nn.Linear(in_ch, joint_ch)
        self.mask_token = nn.Parameter(torch.zeros(joint_ch))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.spatial_blocks = nn.ModuleList([
            SpatialGCNBlock(joint_ch, num_joints, edges, dropout=dropout)
            for _ in range(n_spatial_blocks)
        ])

        self.input_proj = nn.Linear(num_joints * joint_ch, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)

        self.query = nn.Parameter(torch.zeros(num_tokens, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.resampler = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.resampler_norm = nn.LayerNorm(d_model)

        self.out_proj = (nn.Linear(d_model, embed_dim)
                         if embed_dim != d_model else nn.Identity())

        self.d_model = d_model
        self.joint_ch = joint_ch
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def _embed_joints(self, x, joint_frame_mask=None):
        h0 = self.joint_embed(x)  # [B, T, J, joint_ch]
        if joint_frame_mask is not None:
            h0 = torch.where(joint_frame_mask.unsqueeze(-1), self.mask_token, h0)
        h = h0
        for block in self.spatial_blocks:
            h = block(h)
        return h + h0  # stack-level residual, on top of per-block residuals

    def _encode(self, x, mask=None, joint_frame_mask=None):
        b, t, j, c = x.shape
        h = self._embed_joints(x, joint_frame_mask)
        h = h.reshape(b, t, j * self.joint_ch)
        h = self.input_proj(h)
        h = h + sinusoidal_position_encoding(t, self.d_model, x.device, h.dtype)
        key_padding_mask = None if mask is None else ~mask.bool()
        enc = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return enc, key_padding_mask

    def forward_encoder(self, x, mask=None, joint_frame_mask=None):
        """Pretraining path: per-frame hidden state, no pooling."""
        enc, _ = self._encode(x, mask, joint_frame_mask)
        return enc  # [B, T, d_model]

    def forward(self, x, mask=None):
        """Classification path: fixed-size pooled tokens."""
        b = x.shape[0]
        enc, key_padding_mask = self._encode(x, mask)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        pooled, _ = self.resampler(q, enc, enc, key_padding_mask=key_padding_mask)
        pooled = self.resampler_norm(pooled)
        return self.out_proj(pooled)


class ReconstructionDecoder(nn.Module):
    """Pretraining-only scaffolding, discarded before fine-tuning -- does not
    count against the deployed size budget. Maps each frame's temporal hidden
    state back to per-joint xyz (not the presence flag, which isn't a
    reconstruction target)."""

    def __init__(self, d_model, num_joints, hidden=None):
        super().__init__()
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_joints * 3),
        )
        self.num_joints = num_joints

    def forward(self, h):  # h: [B, T, d_model]
        b, t, _ = h.shape
        return self.net(h).reshape(b, t, self.num_joints, 3)


def random_joint_frame_mask(b, t, j, ratio, device):
    """Boolean [B, T, J] mask, True where an entry should be hidden and
    replaced by the learnable mask token."""
    return torch.rand(b, t, j, device=device) < ratio


def build(**kwargs):
    return SkeletonTransformerBackbone(**kwargs)