import torch
import torch.nn as nn

from src.backbones import depth_channel_backbone, skeleton_channel_backbone, thermal_channel_backbone, \
    imu_channel_backbone, ir_channel_backbone, radar_channel_backbone, skeleton_transformer_backbone
from src.backbones import spatial_skeleton, spatialized_skeleton_transformer

BUILDERS = {
    "Depth_Color": depth_channel_backbone.build,
    "IR": ir_channel_backbone.build,
    "Thermal": thermal_channel_backbone.build,
    "IMU": imu_channel_backbone.build,
    "Radar": radar_channel_backbone.build,
    "Skeleton": skeleton_transformer_backbone.build,
}


def param_size_mb(module):
    return sum(p.numel() for p in module.parameters()) * 4 / 1e6


class JointModel(nn.Module):
    """One backbone per modality, each emitting [B, K, Dm] tokens. Tokens are
    projected to a shared fusion width, tagged with learned modality and
    segment position embeddings, and fused by a small transformer with a CLS
    token. Missing or dropped modalities are excluded through the key padding
    mask. Aux heads keep every branch individually discriminative."""

    def __init__(self, modalities, fusion_dim=256, num_tokens=4, n_classes=40,
                 fusion_layers=2, fusion_heads=4, fusion_dropout=0.1,
                 modality_dropout=0.3, backbone_kwargs=None):
        super().__init__()
        self.modalities = list(modalities)
        bk = backbone_kwargs or {}
        self.branches = nn.ModuleDict({
            m: BUILDERS[m](num_tokens=num_tokens, **bk.get(m, {}))
            for m in self.modalities})
        self.projs = nn.ModuleDict({
            m: nn.Linear(self.branches[m].embed_dim, fusion_dim)
            for m in self.modalities})
        self.aux_heads = nn.ModuleDict({
            m: nn.Linear(self.branches[m].embed_dim, n_classes)
            for m in self.modalities})
        self.modality_emb = nn.Parameter(
            torch.zeros(len(self.modalities), fusion_dim))
        self.segment_emb = nn.Parameter(torch.zeros(num_tokens, fusion_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.modality_emb, std=0.02)
        nn.init.trunc_normal_(self.segment_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            fusion_dim, fusion_heads, fusion_dim * 2, dropout=fusion_dropout,
            batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(
            layer, fusion_layers, enable_nested_tensor=False)
        self.fusion_head = nn.Linear(fusion_dim, n_classes)
        self.num_tokens = num_tokens
        self.modality_dropout = modality_dropout

    def _dropped_mask(self, mask):
        """During training, randomly hide present modalities from the fusion
        stage. Keeps at least one modality per sample. Trains robustness to the
        organically missing modalities in the test set."""
        if not self.training or self.modality_dropout <= 0:
            return mask
        drop = (torch.rand_like(mask) < self.modality_dropout).float()
        kept = mask * (1.0 - drop)
        dead = kept.sum(dim=1) == 0
        kept[dead] = mask[dead]
        return kept

    def forward(self, inputs, mask):
        """inputs: dict modality name to tensor. mask: [B, M] presence, 1 if
        the sample has that modality. Branches only run on present rows so that
        zero filled placeholders never pollute BatchNorm statistics."""
        b = mask.size(0)
        fusion_mask = self._dropped_mask(mask)
        tokens, aux = [], {}
        for j, m in enumerate(self.modalities):
            br = self.branches[m]
            t = inputs[m].new_zeros(b, br.num_tokens, br.embed_dim)
            present = mask[:, j] > 0
            if present.any():
                t[present] = br(inputs[m][present])
            aux[m] = self.aux_heads[m](t.mean(dim=1))
            tokens.append(self.projs[m](t)
                          + self.modality_emb[j]
                          + self.segment_emb)
        x = torch.cat(tokens, dim=1)
        pad = fusion_mask.repeat_interleave(self.num_tokens, dim=1) == 0
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        pad = torch.cat(
            [torch.zeros(b, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        x = self.fusion(x, src_key_padding_mask=pad)
        return self.fusion_head(x[:, 0]), aux

    def load_pretrained(self, run_dir, map_location="cpu"):
        import os
        for m in self.modalities:
            path = os.path.join(run_dir, m, "best.pt")
            if not os.path.isfile(path):
                print(f"no checkpoint for {m}, branch stays random")
                continue
            ck = torch.load(path, map_location=map_location)
            self.branches[m].load_state_dict(ck["backbone"])
            self.aux_heads[m].load_state_dict(ck["head"])
            print(f"loaded {m} backbone, val acc {ck.get('val_acc', -1):.4f}")

    def size_report(self):
        total = 0.0
        for m in self.modalities:
            s = param_size_mb(self.branches[m]) + param_size_mb(self.projs[m]) \
                + param_size_mb(self.aux_heads[m])
            total += s
            print(f"branch {m:12s} {s:8.2f} MB")
        fusion = param_size_mb(self.fusion) + param_size_mb(self.fusion_head) \
            + (self.modality_emb.numel() + self.segment_emb.numel()
               + self.cls.numel()) * 4 / 1e6
        total += fusion
        print(f"fusion stage       {fusion:8.2f} MB")
        print(f"total              {total:8.2f} MB")
        return total


if __name__ == "__main__":
    from src.loader.data_loader import MODALITIES, SHAPES
    model = JointModel(MODALITIES)
    model.size_report()
    inputs = {m: torch.randn(2, *SHAPES[m]) for m in MODALITIES}
    mask = torch.tensor([[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 1]],
                        dtype=torch.float32)
    fused, aux = model(inputs, mask)
    print("fused", tuple(fused.shape), "aux keys", list(aux.keys()))
