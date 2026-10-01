import math

import torch
import torch.nn as nn


def sinusoidal_position_encoding(t, d_model, device, dtype=torch.float32):
    """Standard fixed sin/cos positional encoding, computed for whatever
    sequence length is actually passed in. No learned position table and no
    hard-coded max length, so a clip that is longer than anything seen during
    training still gets a sensible (if novel) set of positions instead of
    an out-of-range lookup."""
    position = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype)
                          * (-math.log(10000.0) / d_model))
    pe = torch.zeros(t, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SkeletonTransformerBackbone(nn.Module):
    """Input [B, T, J, 3] plus an optional [B, T] boolean `mask` (True where a
    frame is real, False where it is padding). T can be anything -- short
    clips are simply padded to the batch's max length instead of being
    stretched to a fixed frame count, and long clips are simply longer
    sequences instead of being squeezed down to a fixed count. There is no
    "resample every clip to exactly N frames" step anywhere in this module;
    the only requirement upstream is that everything in one batch shares the
    same T, which a padding collate gives you for free.

    Architecture: per-frame joints are flattened and linearly embedded, a
    small pre-norm Transformer encoder mixes information across time (with
    the padding mask excluding pad frames from attention), and a fixed set of
    `num_tokens` learned query vectors cross-attend into the encoded sequence
    (a small Perceiver-style resampler). That cross-attention is what turns
    "however many valid frames this clip has" into the fixed [num_tokens,
    embed_dim] output every downstream branch/fusion stage expects -- so the
    fixed-size requirement is pushed to the very last step, done by attention
    instead of by frame-index interpolation + jitter.

    Two skip connections on top of the base design:
      - Stack-level residual around the whole temporal encoder: each
        TransformerEncoderLayer already has its own internal residuals, but
        with n_layers=7 the *stack as a whole* can still drift far from the
        embedded input by the final layer. Adding h0 back in lets the encoder
        act as a refinement on top of the per-frame embedding rather than a
        full replacement of it, which is what an increasing depth needs to
        stay easy to optimize.
      - Resampler residual: the learned query vectors have their own
        identity before cross-attending into the sequence; adding that
        identity back after cross-attention lets the resampler refine each
        query toward relevant content instead of having to reconstruct
        useful structure from the attention output alone.

    If `mask` is omitted, every frame in x is treated as valid -- so this
    also works as a drop-in replacement for the old fixed-length pipeline
    while you migrate the loader.
    """

    def __init__(self, num_joints=17, d_model=128, n_layers=13, n_heads=8,
                 ff_mult=2, dropout=0.20, embed_dim=256, num_tokens=4):
        super().__init__()
        self.input_proj = nn.Linear(num_joints * 3, d_model)
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

        self.d_model = d_model
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, x, mask=None):
        b, t, j, c = x.shape
        feats = x.reshape(b, t, j * c)
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


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build()
    m.eval()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,} ({n_params * 4 / 1e6:.3f} MB)")

    # Old-style fixed-length call (no mask) -- still works.
    y = m(torch.randn(2, 64, 17, 3))
    print("fixed-length, no mask:", tuple(y.shape))

    # Variable-length batch: clip A has 40 real frames, clip B has 90. Instead
    # of resampling both to some canonical length, pad both to T=90 and mask
    # out A's tail. No interpolation, no jitter, no upsampling artifacts.
    len_a, len_b = 40, 90
    t_max = max(len_a, len_b)
    x = torch.zeros(2, t_max, 17, 3)
    x[0, :len_a] = torch.randn(len_a, 17, 3)
    x[1, :len_b] = torch.randn(len_b, 17, 3)
    mask = torch.zeros(2, t_max, dtype=torch.bool)
    mask[0, :len_a] = True
    mask[1, :len_b] = True
    y2 = m(x, mask=mask)
    print("variable-length, padded + masked:", tuple(y2.shape))

    # Sanity check: padding a clip further out shouldn't change its output,
    # since the padded frames are masked out of every attention op.
    len_a_padded = t_max + 20
    x_more_pad = torch.zeros(1, len_a_padded, 17, 3)
    x_more_pad[0, :len_a] = x[0, :len_a]
    mask_more_pad = torch.zeros(1, len_a_padded, dtype=torch.bool)
    mask_more_pad[0, :len_a] = True
    y_a_orig = m(x[:1], mask=mask[:1])
    y_a_more_pad = m(x_more_pad, mask=mask_more_pad)
    print("max abs diff from extra padding:",
          (y_a_orig - y_a_more_pad).abs().max().item())