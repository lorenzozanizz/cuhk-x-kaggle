import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0, inplace=True))

    def forward(self, x):
        return self.net(x)


class ConvPatchEmbed(nn.Module):
    """Conv stem, downsamples one frame to a small patch grid."""
    def __init__(self, in_channels, embed_dim, widths=(32, 64, 128)):
        super().__init__()
        blocks, c = [], in_channels
        for w in widths:
            blocks += [ConvBlock2d(c, w, stride=2), ConvBlock2d(w, w)]
            c = w
        self.conv = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(c, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: [B*T, C, H, W] -> [B*T, num_patches, embed_dim]
        f = self.proj(self.conv(x))
        b, d, h, w = f.shape
        return f.flatten(2).transpose(1, 2), (h, w)


class SpatialTransformerBlock(nn.Module):
    """Pre-norm self-attention block over spatial patches within one frame."""
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalQueryPool(nn.Module):
    """Perceiver-style cross-attention pooling: TOKEN_AMT learned queries
    attend over the VISION_FRAMES per-frame features. Fixed input length
    (VISION_FRAMES) so this is a plain cross-attention, no ragged handling
    needed. Learned per-frame-slot positional embedding gives it temporal
    order without assuming any cross-modal correspondence."""
    def __init__(self, dim, num_tokens, num_frames, heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(num_frames, dim) * 0.02)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, frame_feats):
        # frame_feats: [B, T, dim]
        b = frame_feats.shape[0]
        kv = self.norm_kv(frame_feats + self.time_pos.unsqueeze(0))
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm_out(q + out)  # residual around the pooling itself


class ModalityStem(nn.Module):
    """Modality-specific 'conv-transformer': conv patch embed + a couple of
    shallow spatial self-attention blocks per frame, then temporal query
    pooling down to TOKEN_AMT tokens. This is the only part of the model
    that differs per modality."""
    def __init__(self, in_channels, embed_dim, token_amt, num_frames,
                 spatial_depth=2, conv_widths=(32, 64, 128)):
        super().__init__()
        self.patch_embed = ConvPatchEmbed(in_channels, embed_dim, conv_widths)
        self.spatial_pos = None  # lazily sized on first forward (depends on H,W after conv)
        self.spatial_blocks = nn.ModuleList(
            [SpatialTransformerBlock(embed_dim) for _ in range(spatial_depth)])
        self.temporal_pool = TemporalQueryPool(embed_dim, token_amt, num_frames)
        self.embed_dim = embed_dim

    def forward(self, x, mask_ratio=0.0):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        patches, (_, _) = self.patch_embed(x.reshape(b * t, c, h, w))
        if self.spatial_pos is None or self.spatial_pos.shape[1] != patches.shape[1]:
            self.spatial_pos = nn.Parameter(
                torch.randn(1, patches.shape[1], self.embed_dim, device=x.device) * 0.02)
        patches = patches + self.spatial_pos

        if mask_ratio > 0.0:
            # random patch masking for MAE-style pretraining
            n = patches.shape[1]
            keep = int(n * (1 - mask_ratio))
            noise = torch.rand(patches.shape[0], n, device=x.device)
            keep_idx = noise.argsort(dim=1)[:, :keep]
            patches = torch.gather(
                patches, 1, keep_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        for blk in self.spatial_blocks:
            patches = blk(patches)

        frame_feat = patches.mean(dim=1).reshape(b, t, self.embed_dim)  # [B,T,D]
        return self.temporal_pool(frame_feat)  # [B, TOKEN_AMT, D]


class SharedBackbone(nn.Module):
    """Weight-shared across all three modalities. Pre-norm residual transformer
    blocks over the TOKEN_AMT tokens. Instantiate ONCE, pass the same instance
    into every ModalityBackbone below."""
    def __init__(self, embed_dim, depth=3, heads=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SpatialTransformerBlock(embed_dim, heads=heads) for _ in range(depth)])

    def forward(self, tokens):
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens


class ModalityBackbone(nn.Module):
    """Drop-in replacement for the old DepthColorBackbone. Same external
    interface: forward(x) -> [B, TOKEN_AMT, embed_dim]. Internally: modality
    stem -> shared backbone -> residual. Handles missing modalities via a
    learned 'absent' embedding instead of encoding an all-zero clip."""
    def __init__(self, in_channels, embed_dim, token_amt, num_frames, shared_backbone):
        super().__init__()
        self.stem = ModalityStem(in_channels, embed_dim, token_amt, num_frames)
        self.shared = shared_backbone
        self.absent_tokens = nn.Parameter(torch.randn(token_amt, embed_dim) * 0.02)
        self.embed_dim = embed_dim
        self.num_tokens = token_amt

    def forward(self, x, present=None, mask_ratio=0.0):
        # x: [B, T, C, H, W], present: [B] bool or None (assume all present)
        b = x.shape[0]
        stem_out = self.stem(x, mask_ratio=mask_ratio)          # [B, TOKEN_AMT, D]
        out = stem_out + self.shared(stem_out)                   # residual around shared block
        if present is not None:
            absent = self.absent_tokens.unsqueeze(0).expand(b, -1, -1)
            present_mask = present.view(b, 1, 1).to(out.dtype)
            out = out * present_mask + absent * (1 - present_mask)
        return out

class PretrainDecoder(nn.Module):
    """Reconstructs the VISION_FRAMES x C x H x W clip from the TOKEN_AMT
    embedding. Only used during self-supervised pretraining; never touches
    the external interface, never gets saved into the final checkpoint."""
    def __init__(self, embed_dim, out_channels, num_frames, out_size, up_widths=(128, 64, 32)):
        super().__init__()
        self.num_frames = num_frames
        self.frame_query = nn.Parameter(torch.randn(num_frames, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        seed = out_size // (2 ** len(up_widths))
        self.seed = seed
        self.to_map = nn.Linear(embed_dim, up_widths[0] * seed * seed)
        ups, c = [], up_widths[0]
        for w in up_widths[1:]:
            ups += [nn.ConvTranspose2d(c, w, 4, stride=2, padding=1), nn.GELU()]
            c = w
        ups += [nn.ConvTranspose2d(c, out_channels, 4, stride=2, padding=1)]
        self.up = nn.Sequential(*ups)
        self.embed_dim = embed_dim

    def forward(self, tokens):
        b = tokens.shape[0]
        q = self.frame_query.unsqueeze(0).expand(b, -1, -1)
        f, _ = self.cross_attn(q, tokens, tokens, need_weights=False)
        f = self.norm(f)                                   # [B, T, D]
        f = self.to_map(f).reshape(b * self.num_frames, -1, self.seed, self.seed)
        rec = self.up(f)
        c, h, w = rec.shape[1], rec.shape[2], rec.shape[3]
        return rec.reshape(b, self.num_frames, c, h, w)


def pretrain_step(stem_backbone, decoder, x, mask_ratio=0.6):
    """Denoising / masked-autoencoder step for one modality."""
    tokens = stem_backbone(x, mask_ratio=mask_ratio)
    recon = decoder(tokens)
    return F.mse_loss(recon, x)