"""Trains one vision modality backbone (stem + shared backbone) with either
a masked-reconstruction pretraining objective or a linear-probe classifier
head on mean-pooled tokens.

Run once per modality, per mode. The shared backbone is persisted separately
from the modality stem so that successive runs (across different modalities,
and across pretrain -> probe) continue updating the same shared weights
rather than reinitializing them.

Outputs per run: runs/<MODALITY>/<MODE>/best.pt, history.json, curves.png
Shared backbone checkpoint: runs/shared_backbone.pt (updated in place)
"""

import os
import json
import random
import numpy as np
import torch
import tqdm
from collections import Counter

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.loader.data_loader import (DataIndex, split_by_user, split_random,
                                     SingleModalityDataset, NUM_CLASSES)
from loader.vision_loader import VISION_FRAMES, VISION_SIZE
from backbones.vision_backbone import (SharedBackbone, ModalityBackbone,
                                        PretrainDecoder)

def param_size_mb(model, bytes_per_param=4):
    """bytes_per_param=4 for fp32, 2 for fp16/bf16."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * bytes_per_param / (1024 ** 2)
DATA_ROOT = "C:/Users/picul/Documents/Research/Kaggle/UHK-X/Training/data/HAR/data/"
CACHE_ROOT = "C:/Users/picul/Documents/Research/Kaggle/UHK-X/Training/data/HAR/data/cache/"

MODALITY = "Depth_Color"     # "Depth_Color" | "IR" | "Thermal"  (keys of Loaders.get)
MODE = "probe"                # "pretrain" | "probe"
RUN_DIR = "runs"
SHARED_BACKBONE_PATH = os.path.join(RUN_DIR, "shared_backbone.pt")
SEED = 0
VAL_FRACTION = 0.2
SPLIT_BY_USER = True
EPOCHS = 25
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 2
NUM_TOKENS = 7                # must match TOKEN_AMT used by the joint model
EMBED_DIM = 128
MASK_RATIO = 0.65             # pretrain only
FREEZE_BACKBONE_IN_PROBE = False   # True = pure linear probe, False = fine-tune
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODALITY_CHANNELS = {"Depth_Color": 3, "IR": 1, "Thermal": 3}


# ------------------------------------------------------------- backbone ----

def build_backbone(modality, num_tokens, embed_dim):
    """Loads the shared backbone from disk if present (continuing whatever
    state prior runs -- other modalities, or a prior pretrain phase -- left
    it in), otherwise initializes fresh. Returns (modality_backbone, shared)
    so the shared module can be saved back out after training."""
    shared = SharedBackbone(embed_dim, depth=3, heads=4)
    if os.path.exists(SHARED_BACKBONE_PATH):
        shared.load_state_dict(torch.load(SHARED_BACKBONE_PATH, map_location="cpu"))
        print(f"loaded shared backbone from {SHARED_BACKBONE_PATH}")
    else:
        print("no shared backbone checkpoint found, starting fresh")

    channels = MODALITY_CHANNELS[modality]
    backbone = ModalityBackbone(channels, embed_dim, num_tokens, VISION_FRAMES, shared)

    stem_ckpt = os.path.join(RUN_DIR, modality, "pretrain", "stem.pt")
    if MODE == "probe" and os.path.exists(stem_ckpt):
        backbone.stem.load_state_dict(torch.load(stem_ckpt, map_location="cpu"))
        print(f"loaded pretrained stem from {stem_ckpt}")

    return backbone, shared


def save_shared(shared):
    os.makedirs(RUN_DIR, exist_ok=True)
    torch.save(shared.state_dict(), SHARED_BACKBONE_PATH)


# ---------------------------------------------------------------- probe ----

class BackboneClassifier(nn.Module):
    def __init__(self, backbone, n_classes, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, n_classes)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            feats = self.backbone(x)
        return self.head(feats.mean(dim=1))


def run_epoch_probe(model, loader, loss_fn, opt=None):
    training = opt is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in tqdm.tqdm(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


# -------------------------------------------------------------- pretrain ---
# Pretraining needs clips only, no labels. SingleModalityDataset yields
# (x, label) pairs, so wrap it and drop the label rather than writing a
# separate dataset class.

class UnlabeledWrapper(torch.utils.data.Dataset):
    def __init__(self, labeled_ds):
        self.ds = labeled_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, _ = self.ds[i]
        return x


def run_epoch_pretrain(backbone, decoder, loader, opt=None):
    training = opt is not None
    backbone.train(training)
    decoder.train(training)
    total, loss_sum = 0, 0.0
    with torch.set_grad_enabled(training):
        for x in tqdm.tqdm(loader):
            x = x.to(DEVICE)
            tokens = backbone.stem(x, mask_ratio=MASK_RATIO)
            tokens = tokens + backbone.shared(tokens)   # same path as inference, minus absent-mask
            recon = decoder(tokens)
            loss = F.mse_loss(recon, x)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
    return loss_sum / max(total, 1)


# ---------------------------------------------------------------- plots ----

def plot_history(hist, out_path, mode):
    if mode == "probe":
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.plot(hist["train_loss"], label="train")
        a1.plot(hist["val_loss"], label="val")
        a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.legend()
        a2.plot(hist["val_acc"], label="val acc")
        a2.axhline(1.0 / NUM_CLASSES, color="gray", ls="--", label="chance")
        a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.legend()
    else:
        fig, a1 = plt.subplots(1, 1, figsize=(6, 4))
        a1.plot(hist["train_loss"], label="train")
        a1.plot(hist["val_loss"], label="val")
        a1.set_xlabel("epoch"); a1.set_ylabel("recon MSE"); a1.legend()
    fig.suptitle(f"{hist['modality']} [{mode}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ main --

def main():
    print(DEVICE, MODALITY, MODE)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    data_index = DataIndex(DATA_ROOT, CACHE_ROOT)
    data_index.statistics()
    samples = data_index.get_samples()
    print(f"indexed {len(samples)} samples")

    if SPLIT_BY_USER:
        train_s, val_s, val_users = split_by_user(samples, VAL_FRACTION, SEED)
        print("validation users:", val_users)
    else:
        train_s, val_s = split_random(samples, VAL_FRACTION, SEED)

    # SingleModalityDataset already filters to samples that have this
    # modality (checks `modality in s["paths"]`) and calls Loaders.get(modality).
    train_ds = SingleModalityDataset(train_s, MODALITY, train=True)
    val_ds = SingleModalityDataset(val_s, MODALITY, train=False)
    print(f"{MODALITY}: {len(train_ds)} train, {len(val_ds)} val clips")

    if MODE == "pretrain":
        train_ds = UnlabeledWrapper(train_ds)
        val_ds = UnlabeledWrapper(val_ds)

    train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, drop_last=True)
    val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    backbone, shared = build_backbone(MODALITY, NUM_TOKENS, EMBED_DIM)

    out_dir = os.path.join(RUN_DIR, MODALITY, MODE)
    os.makedirs(out_dir, exist_ok=True)

    if MODE == "pretrain":
        decoder = PretrainDecoder(EMBED_DIM, MODALITY_CHANNELS[MODALITY],
                                   VISION_FRAMES, VISION_SIZE).to(DEVICE)
        backbone = backbone.to(DEVICE)
        size_mb = param_size_mb(backbone) + param_size_mb(decoder)
        print(f"stem+shared {param_size_mb(backbone):.2f} MB "
              f"(decoder, discarded, {param_size_mb(decoder):.2f} MB)")

        opt = torch.optim.AdamW(
            list(backbone.parameters()) + list(decoder.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

        hist = {"modality": MODALITY, "mode": MODE, "seed": SEED,
                "train_loss": [], "val_loss": [], "best_val_loss": float("inf")}

        for epoch in tqdm.tqdm(range(EPOCHS)):
            tr_loss = run_epoch_pretrain(backbone, decoder, train_ld, opt)
            va_loss = run_epoch_pretrain(backbone, decoder, val_ld)
            sched.step()
            hist["train_loss"].append(tr_loss)
            hist["val_loss"].append(va_loss)
            if va_loss < hist["best_val_loss"]:
                hist["best_val_loss"] = va_loss
                torch.save(backbone.stem.state_dict(), os.path.join(out_dir, "stem.pt"))
                save_shared(shared)   # <-- persists across modalities/runs
                torch.save({"val_loss": va_loss, "num_tokens": NUM_TOKENS},
                           os.path.join(out_dir, "best.pt"))
            print(f"epoch {epoch+1:3d} train {tr_loss:.5f} val {va_loss:.5f} "
                  f"best {hist['best_val_loss']:.5f}")

    else:  # probe
        model = BackboneClassifier(backbone, NUM_CLASSES,
                                    freeze_backbone=FREEZE_BACKBONE_IN_PROBE).to(DEVICE)
        size_mb = param_size_mb(model)
        print(f"backbone {param_size_mb(model.backbone):.2f} MB, "
              f"head {param_size_mb(model.head):.2f} MB, total {size_mb:.2f} MB")

        counts = Counter(entry["label"] for entry in train_ds.items)
        weights = torch.tensor([1.0 / counts.get(c, 1) for c in range(NUM_CLASSES)])
        weights = weights / weights.sum() * NUM_CLASSES
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=LABEL_SMOOTHING)

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

        hist = {"modality": MODALITY, "mode": MODE, "size_mb": size_mb, "seed": SEED,
                "split_by_user": SPLIT_BY_USER, "train_loss": [], "val_loss": [],
                "val_acc": [], "best_val_acc": 0.0}

        for epoch in tqdm.tqdm(range(EPOCHS)):
            tr_loss, tr_acc = run_epoch_probe(model, train_ld, loss_fn, opt)
            va_loss, va_acc = run_epoch_probe(model, val_ld, loss_fn)
            sched.step()
            hist["train_loss"].append(tr_loss)
            hist["val_loss"].append(va_loss)
            hist["val_acc"].append(va_acc)
            if va_acc > hist["best_val_acc"]:
                hist["best_val_acc"] = va_acc
                torch.save({"backbone": model.backbone.state_dict(),
                            "head": model.head.state_dict(),
                            "val_acc": va_acc, "num_tokens": NUM_TOKENS},
                           os.path.join(out_dir, "best.pt"))
                if not FREEZE_BACKBONE_IN_PROBE:
                    save_shared(shared)   # fine-tuning also updates shared weights
            print(f"epoch {epoch+1:3d}  train loss {tr_loss:.4f} acc {tr_acc:.4f}"
                  f"  val loss {va_loss:.4f} acc {va_acc:.4f}"
                  f"  best {hist['best_val_acc']:.4f}")

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist, os.path.join(out_dir, "curves.png"), MODE)
    print(f"done, artifacts in {out_dir}")


if __name__ == "__main__":
    main()