import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def build_skeleton_adjacency(edges=SKELETON_EDGES, num_joints=17,
                              symmetric_norm=True, dtype=torch.float32):
    """Builds the normalized adjacency matrix A_hat used by SpatialGraphConv.

    Standard Kipf & Welling GCN normalization: A_hat = D^-1/2 (A + I) D^-1/2,
    where A is the (symmetrized) skeleton bone adjacency and I adds self-loops
    so each joint's own features always pass through. Symmetric normalization
    keeps the operator's eigenvalues in a well-behaved range regardless of how
    unevenly connected individual joints are (e.g. joint 8, the chest/neck
    hub, has degree 4 vs an end-effector joint like 3 or 16 with degree 1) --
    without it, high-degree joints get systematically over/under-weighted.

    Args:
        edges: list of (i, j) bone connections, 0-indexed, undirected.
        num_joints: number of joints/nodes.
        symmetric_norm: if False, returns the raw A + I (unnormalized).
        dtype: floating point dtype for the returned tensor.

    Returns:
        A_hat: [num_joints, num_joints] tensor.
    """
    A = torch.zeros(num_joints, num_joints, dtype=dtype)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0  # skeleton bones are undirected for this purpose
    A = A + torch.eye(num_joints, dtype=dtype)  # self-loops

    if not symmetric_norm:
        return A

    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat


# ---------------------------------------------------------------------------
# Spatial graph convolution
# ---------------------------------------------------------------------------

class SpatialGraphConv(nn.Module):
    """One graph-conv layer applied independently at every frame:
    for each frame, mix joint features along the skeleton graph (A_hat @ X),
    then project channels (@ W). This is what turns "17 independent numbers"
    into "17 joints that know about their physically connected neighbors"
    before anything ever gets flattened into a single per-frame vector.

    A_hat is the fixed anatomical prior (registered as a buffer -- no
    gradient, doesn't count as a parameter, moves with .to(device) and shows
    up in state_dict so a loaded checkpoint always has the topology it was
    trained with). `edge_gate` is a small learnable additive delta on top of
    it, tanh-bounded and scaled down, so the model can *slightly* reweight or
    add long-range edges (e.g. hand-to-hand coordination that isn't a real
    bone) without being able to override the skeleton structure wholesale --
    that bounded-perturbation design is a deliberately shrunk version of what
    CTR-GCN's channel-wise topology refinement does, sized for a much smaller
    parameter budget (one shared [J,J] gate here vs. per-channel topologies).

    Args:
        num_joints: J.
        in_c: input channels per joint.
        out_c: output channels per joint.
        A_hat: [J, J] normalized adjacency from build_skeleton_adjacency.
        gate_scale: max magnitude of the learnable perturbation to A_hat.
    """

    def __init__(self, num_joints, in_c, out_c, A_hat, gate_scale=0.1):
        super().__init__()
        assert A_hat.shape == (num_joints, num_joints)
        self.register_buffer("A_hat", A_hat)
        self.edge_gate = nn.Parameter(torch.zeros(num_joints, num_joints))
        self.gate_scale = gate_scale
        self.lin = nn.Linear(in_c, out_c)
        self.norm = nn.LayerNorm(out_c)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        # x: [B, T, J, C_in] -> [B, T, J, C_out]
        A = self.A_hat + self.gate_scale * torch.tanh(self.edge_gate)
        x = torch.einsum("ij,btjc->btic", A, x)
        x = self.lin(x)
        return self.act(self.norm(x) + x)


class SkeletonGCNStem(nn.Module):
    """Small stack of SpatialGraphConv layers, with a residual connection
    around the stack for the same reason the Transformer encoder gets one:
    depth should refine the per-joint features, not have to reconstruct them
    from scratch. Output is still [B, T, J, C] -- flattening to [B, T, J*C]
    for the Transformer's input_proj happens outside this module, in the
    backbone, so this stem is reusable independent of what follows it.
    """

    def __init__(self, num_joints, in_c, hidden_c, n_layers, A_hat,
                 gate_scale=0.1):
        super().__init__()
        assert n_layers >= 1
        layers = [SpatialGraphConv(num_joints, in_c, hidden_c, A_hat, gate_scale)]
        for _ in range(n_layers - 1):
            layers.append(
                SpatialGraphConv(num_joints, hidden_c, hidden_c, A_hat, gate_scale))
        self.layers = nn.ModuleList(layers)
        # project the raw input to hidden_c so the stack-level residual is
        # shape-compatible even though in_c != hidden_c on the first layer
        self.in_proj = (nn.Linear(in_c, hidden_c) if in_c != hidden_c
                         else nn.Identity())

    def forward(self, x):
        h0 = self.in_proj(x)
        h = x
        for layer in self.layers:
            h = layer(h)
        return h + h0  # stack-level residual, shapes: [B, T, J, hidden_c]


# ---------------------------------------------------------------------------
# Positional encoding (unchanged)
# ---------------------------------------------------------------------------

def sinusoidal_position_encoding(t, d_model, device, dtype=torch.float32):
    """Standard fixed sin/cos positional encoding, computed for whatever
    sequence length is actually passed in."""
    position = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype)
                          * (-math.log(10000.0) / d_model))
    pe = torch.zeros(t, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class SkeletonTransformerBackbone(nn.Module):
    """Input [B, T, J, 3] plus an optional [B, T] boolean `mask` (True where a
    frame is real, False where it is padding).

    Pipeline:
      1. SkeletonGCNStem: per-frame graph convolution over the skeleton
         adjacency, turning raw joint coordinates into joint features that
         are aware of their physical neighbors. Runs independently per frame
         (no temporal mixing here), so it's unaffected by padding/mask.
      2. Joint-identity embedding: a learnable per-joint vector added after
         the GCN stem, so the model can distinguish "this is joint 8" (the
         chest/neck hub) from "this is joint 3" (an end effector) beyond
         whatever the graph topology already implies. Cheap (J x C params)
         and orthogonal to the temporal positional encoding below.
      3. Flatten joints, linear-embed to d_model, add temporal sin/cos
         positional encoding, run through the pre-norm Transformer encoder
         (padding mask excludes pad frames from attention), stack-level
         residual.
      4. Perceiver-style resampler: fixed learned queries cross-attend into
         the encoded sequence to produce the fixed [num_tokens, embed_dim]
         output, with a resampler residual.

    If `mask` is omitted, every frame in x is treated as valid.
    """

    def __init__(self, num_joints=17, d_model=128, n_layers=12, n_heads=8,
                 ff_mult=2, dropout=0.30, embed_dim=256, num_tokens=4,
                 gcn_hidden=16, gcn_layers=1, gcn_gate_scale=0.1,
                 edges=SKELETON_EDGES):
        super().__init__()

        A_hat = build_skeleton_adjacency(edges, num_joints)
        self.gcn_stem = SkeletonGCNStem(
            num_joints, in_c=3, hidden_c=gcn_hidden, n_layers=gcn_layers,
            A_hat=A_hat, gate_scale=gcn_gate_scale)

        self.joint_embed = nn.Parameter(torch.zeros(num_joints, gcn_hidden))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

        self.input_proj = nn.Linear(num_joints * gcn_hidden, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(d_model)

        self.query = nn.Parameter(torch.zeros(num_tokens, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.resampler = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.resampler_norm = nn.LayerNorm(d_model)

        self.out_proj = (nn.Linear(d_model, embed_dim)
                         if embed_dim != d_model else nn.Identity())

        self.num_joints = num_joints
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.gcn_hidden = gcn_hidden

    def forward(self, x, mask=None):
        b, t, j, c = x.shape
        assert j == self.num_joints, f"expected {self.num_joints} joints, got {j}"

        g = self.gcn_stem(x)                       # [B, T, J, gcn_hidden]
        g = g + self.joint_embed.view(1, 1, j, -1)  # per-joint identity

        feats = g.reshape(b, t, j * self.gcn_hidden)
        h0 = self.input_proj(feats)
        h0 = h0 + sinusoidal_position_encoding(t, self.d_model, x.device, h0.dtype)

        key_padding_mask = None if mask is None else ~mask.bool()
        enc = self.encoder(h0, src_key_padding_mask=key_padding_mask)
        enc = self.encoder_norm(enc + h0)  # stack-level residual

        q = self.query.unsqueeze(0).expand(b, -1, -1)
        pooled, _ = self.resampler(q, enc, enc,
                                    key_padding_mask=key_padding_mask)
        pooled = self.resampler_norm(pooled + q)  # resampler residual
        return self.out_proj(pooled)


def build(**kwargs):
    return SkeletonTransformerBackbone(**kwargs)