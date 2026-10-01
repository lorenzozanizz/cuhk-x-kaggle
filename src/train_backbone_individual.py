"""Trains one modality backbone with a linear classifier head on mean pooled
tokens. Edit the globals below, run once per modality. Outputs per run:
runs/<MODALITY>/best.pt, history.json, curves.png"""

import os
import json
import random
import numpy as np
import torch
import tqdm
from collections import Counter

import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from loader import load_radar
from src.loader.data_loader import (DataIndex, split_by_user, split_random,
                         SingleModalityDataset, NUM_CLASSES)
from src.loader.standardizer import Standardizer
from src.joint_model import BUILDERS, param_size_mb

from dotenv import load_dotenv

MODALITY = "Skeleton"
RUN_DIR = "runs"
SEED = 0
VAL_FRACTION = 0.2
SPLIT_BY_USER = True
EPOCHS = 33
BATCH_SIZE = 12
LR = 1.5e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 2
NUM_TOKENS = 5
BACKBONE_KWARGS = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class BackboneClassifier(nn.Module):

    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x).mean(dim=1))


def run_epoch(model, loader, loss_fn, opt=None, sched=None, clip_norm=1.0):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                opt.step()
                sched.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
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
    fig.suptitle(hist["modality"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

import math

def build_scheduler(opt, steps_per_epoch, epochs, warmup_frac=0.05, min_lr_frac=0.01):
    """Linear warmup then cosine decay to min_lr_frac * base_lr, stepped
    per-batch. warmup_frac=0.05 with ~25 epochs and a real steps_per_epoch
    gives a smooth ramp instead of the 1-2 discrete jumps you'd get stepping
    a warmup schedule at epoch granularity."""
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def build_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def main():
    load_dotenv()

    DATA_ROOT = os.getenv('DATASET_PATH')
    CACHE_ROOT = os.getenv('CACHE_PATH')

    print(DEVICE)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Construct the data index, then get the indiivdual samples ( e.g. triples for user actions )
    data_index = DataIndex(DATA_ROOT, CACHE_ROOT)
    data_index.statistics()
    samples = data_index.get_samples()

    print(f"indexed {len(samples)} samples")
    if SPLIT_BY_USER:
        train_s, val_s, val_users = split_by_user(samples, VAL_FRACTION, SEED)
        print("validation users:", val_users)
    else:
        train_s, val_s = split_random(samples, VAL_FRACTION, SEED)

    # Channel statistics are fitted on the training split only. Delete the
    # json when changing SEED, VAL_FRACTION or loader constants.
    Standardizer.ensure(train_s)


    train_ds = SingleModalityDataset(train_s, MODALITY, train=True)
    val_ds = SingleModalityDataset(val_s, MODALITY, train=False)
    print(f"{MODALITY}: {len(train_ds)} train, {len(val_ds)} val clips")
    train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
    val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)

    backbone = BUILDERS[MODALITY](num_tokens=NUM_TOKENS, **BACKBONE_KWARGS)
    model = BackboneClassifier(backbone, NUM_CLASSES).to(DEVICE)
    size_mb = param_size_mb(model)
    print(f"backbone {param_size_mb(model.backbone):.2f} MB, "
          f"head {param_size_mb(model.head):.2f} MB, total {size_mb:.2f} MB")

    counts = Counter(entry['label'] for entry in train_s)  # adjust to your sample->label access
    weights = torch.tensor([1.0 / counts[c] for c in range(NUM_CLASSES)])
    weights = weights / weights.sum() * NUM_CLASSES
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=LABEL_SMOOTHING)

    opt = torch.optim.AdamW(build_param_groups(model, WEIGHT_DECAY),
                            lr=LR, weight_decay=WEIGHT_DECAY)  # per-group WD overrides this

    # opt = torch.optim.AdamW(model.parameters(), lr=LR,                       weight_decay=WEIGHT_DECAY)

    sched = build_scheduler(opt, len(train_ld), EPOCHS)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    out_dir = os.path.join(RUN_DIR, MODALITY)
    os.makedirs(out_dir, exist_ok=True)
    hist = {"modality": MODALITY, "size_mb": size_mb, "seed": SEED,
            "split_by_user": SPLIT_BY_USER, "train_loss": [], "val_loss": [],
            "val_acc": [], "best_val_acc": 0.0}

    for epoch in tqdm.tqdm(range(EPOCHS)):
        tr_loss, tr_acc = run_epoch(model, train_ld, loss_fn, opt, sched=sched)
        va_loss, va_acc = run_epoch(model, val_ld, loss_fn)
        sched.step()
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        if va_acc > hist["best_val_acc"]:
            hist["best_val_acc"] = va_acc
            torch.save({"backbone": model.backbone.state_dict(),
                        "head": model.head.state_dict(),
                        "val_acc": va_acc,
                        "num_tokens": NUM_TOKENS,
                        "backbone_kwargs": BACKBONE_KWARGS},
                       os.path.join(out_dir, "best.pt"))
        print(f"epoch {epoch + 1:3d}  train loss {tr_loss:.4f} acc {tr_acc:.4f}"
              f"  val loss {va_loss:.4f} acc {va_acc:.4f}"
              f"  best {hist['best_val_acc']:.4f}")

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist, os.path.join(out_dir, "curves.png"))
    print(f"done, best val acc {hist['best_val_acc']:.4f}, "
          f"artifacts in {out_dir}")


if __name__ == "__main__":
    main()
