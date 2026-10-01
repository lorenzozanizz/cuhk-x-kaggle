"""Compares the individually trained backbones from their saved histories.
Prints a table with size and accuracy, flags branches that are not learning,
and writes runs/comparison.png with validation curves and best accuracies."""

import os
import json
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.loader.data_loader import NUM_CLASSES

RUN_DIR = "../runs"
CHANCE = 1.0 / NUM_CLASSES
FAIL_FACTOR = 2.0


def load_histories(run_dir):
    hists = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "history.json"))):
        try:
            with open(path) as f:
                hists.append(json.load(f))
        except Exception as e:
            print(f"could not read {path}: {e}")
    return hists


def status(h):
    best = h["best_val_acc"]
    if best < FAIL_FACTOR * CHANCE:
        return "NOT LEARNING"
    if h["val_loss"] and h["train_loss"]:
        gap = h["val_loss"][-1] - h["train_loss"][-1]
        if gap > 1.5:
            return "OVERFITTING"
    return "OK"


def main():
    hists = load_histories(RUN_DIR)
    if not hists:
        print(f"no histories found under {RUN_DIR}")
        return
    hists.sort(key=lambda h: h["best_val_acc"], reverse=True)

    header = (f"{'modality':14s} {'MB':>8s} {'best acc':>9s} "
              f"{'tr loss':>8s} {'va loss':>8s} {'epochs':>7s} status")
    print(header)
    print("-" * len(header))
    for h in hists:
        print(f"{h['modality']:14s} {h['size_mb']:8.2f} "
              f"{h['best_val_acc']:9.4f} {h['train_loss'][-1]:8.4f} "
              f"{h['val_loss'][-1]:8.4f} {len(h['val_acc']):7d} {status(h)}")
    print(f"chance level: {CHANCE:.4f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for h in hists:
        a1.plot(h["val_acc"], label=h["modality"])
    a1.axhline(CHANCE, color="gray", ls="--", label="chance")
    a1.set_xlabel("epoch")
    a1.set_ylabel("validation accuracy")
    a1.legend(fontsize=8)

    names = [h["modality"] for h in hists]
    accs = [h["best_val_acc"] for h in hists]
    sizes = [h["size_mb"] for h in hists]
    bars = a2.bar(names, accs)
    for bar, s in zip(bars, sizes):
        a2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s:.1f}MB", ha="center", va="bottom", fontsize=8)
    a2.axhline(CHANCE, color="gray", ls="--")
    a2.set_ylabel("best validation accuracy")
    a2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out = os.path.join(RUN_DIR, "comparison.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
