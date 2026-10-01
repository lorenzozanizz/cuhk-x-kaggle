"""Trains the joint multimodal model. Two stages controlled by STAGE:
  fusion:   backbones frozen (loaded from individual runs), fusion trains
  finetune: everything trains, backbones at a much lower learning rate
Run fusion first, then finetune starting from the fusion checkpoint."""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import (build_index, split_by_user, split_random,
                         MultiModalityDataset, NUM_CLASSES,
                         load_stats, fit_stats)
from joint_model import JointModel

DATA_ROOT = "HAR/data"
MODALITIES = ["Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton"]
RUN_DIR = "runs"
OUT_DIR = "runs/joint"
STAGE = "fusion"
LOAD_BACKBONES = True
RESUME_JOINT = None
SEED = 0
VAL_FRACTION = 0.2
SPLIT_BY_USER = True
EPOCHS = 30
BATCH_SIZE = 16
LR_FUSION = 3e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
AUX_WEIGHT = 0.3
MODALITY_DROPOUT = 0.3
FUSION_DIM = 256
NUM_TOKENS = 4
FUSION_LAYERS = 2
FUSION_HEADS = 4
GRAD_CLIP = 1.0
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def aux_loss(aux, mask, y, loss_fn):
    """Cross entropy per modality, restricted to samples where the modality is
    actually present, averaged over contributing modalities."""
    total, count = 0.0, 0
    for j, m in enumerate(MODALITIES):
        pm = mask[:, j] > 0
        if pm.any():
            total = total + loss_fn(aux[m][pm], y[pm])
            count += 1
    return total / max(count, 1)


def run_epoch(model, loader, loss_fn, opt=None):
    training = opt is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for inputs, mask, y in loader:
            inputs = {m: v.to(DEVICE) for m, v in inputs.items()}
            mask, y = mask.to(DEVICE), y.to(DEVICE)
            fused, aux = model(inputs, mask)
            loss = loss_fn(fused, y) + AUX_WEIGHT * aux_loss(aux, mask, y,
                                                             loss_fn)
            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (fused.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def plot_history(hist, out_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["train_loss"], label="train")
    a1.plot(hist["val_loss"], label="val")
    a1.set_xlabel("epoch")
    a1.set_ylabel("loss")
    a1.legend()
    a2.plot(hist["val_acc"], label="val acc")
    a2.axhline(1.0 / NUM_CLASSES, color="gray", ls="--", label="chance")
    a2.set_xlabel("epoch")
    a2.set_ylabel("accuracy")
    a2.legend()
    fig.suptitle(f"joint model, stage {hist['stage']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    samples = build_index(DATA_ROOT)
    print(f"indexed {len(samples)} samples")
    if SPLIT_BY_USER:
        train_s, val_s, val_users = split_by_user(samples, VAL_FRACTION, SEED)
        print("validation users:", val_users)
    else:
        train_s, val_s = split_random(samples, VAL_FRACTION, SEED)

    if load_stats() is None:
        fit_stats(train_s)

    train_ds = MultiModalityDataset(train_s, MODALITIES, train=True)
    val_ds = MultiModalityDataset(val_s, MODALITIES, train=False)
    print(f"{len(train_ds)} train, {len(val_ds)} val clips")
    train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
    val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)

    model = JointModel(MODALITIES, fusion_dim=FUSION_DIM,
                       num_tokens=NUM_TOKENS, n_classes=NUM_CLASSES,
                       fusion_layers=FUSION_LAYERS, fusion_heads=FUSION_HEADS,
                       modality_dropout=MODALITY_DROPOUT).to(DEVICE)
    if RESUME_JOINT:
        model.load_state_dict(torch.load(RESUME_JOINT,
                                         map_location=DEVICE)["model"])
        print(f"resumed joint weights from {RESUME_JOINT}")
    elif LOAD_BACKBONES:
        model.load_pretrained(RUN_DIR, map_location=DEVICE)
    model.size_report()

    backbone_params, fusion_params = [], []
    for name, p in model.named_parameters():
        (backbone_params if name.startswith("branches.")
         else fusion_params).append(p)
    if STAGE == "fusion":
        for p in backbone_params:
            p.requires_grad = False
        groups = [{"params": fusion_params, "lr": LR_FUSION}]
    else:
        groups = [{"params": fusion_params, "lr": LR_FUSION},
                  {"params": backbone_params, "lr": LR_BACKBONE}]
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    os.makedirs(OUT_DIR, exist_ok=True)
    hist = {"stage": STAGE, "seed": SEED, "modalities": MODALITIES,
            "train_loss": [], "val_loss": [], "val_acc": [],
            "best_val_acc": 0.0}

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_ld, loss_fn, opt)
        va_loss, va_acc = run_epoch(model, val_ld, loss_fn)
        sched.step()
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        if va_acc > hist["best_val_acc"]:
            hist["best_val_acc"] = va_acc
            torch.save({"model": model.state_dict(), "val_acc": va_acc,
                        "stage": STAGE},
                       os.path.join(OUT_DIR, f"best_{STAGE}.pt"))
        print(f"epoch {epoch + 1:3d}  train loss {tr_loss:.4f} acc {tr_acc:.4f}"
              f"  val loss {va_loss:.4f} acc {va_acc:.4f}"
              f"  best {hist['best_val_acc']:.4f}")

    with open(os.path.join(OUT_DIR, f"history_{STAGE}.json"), "w") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist, os.path.join(OUT_DIR, f"curves_{STAGE}.png"))
    print(f"done, best val acc {hist['best_val_acc']:.4f}, "
          f"artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
