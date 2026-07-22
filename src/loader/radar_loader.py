from .dataset_config import *
from .loading_utils import read_csv_any

import os
import glob

import numpy as np
import torch


def load_radar(tdir, train):
    """Point-level radar loader.

    Previous version collapsed each frame's 5-10 points to mean/std/max
    summary stats before any learnable layer saw them -- with this few
    points per frame, std/max are mostly noise, and the collapse happens
    *before* the network can learn to tell signal from noise.

    Instead: keep raw points, bin frames into RADAR_LEN temporal groups
    (concatenating all points from all raw frames that fall in a bin --
    this matters because we have hundreds of raw frames and don't want
    to just drop most of them via naive interpolation/nearest-frame
    sampling), then pad/truncate each bin to RADAR_MAX_PTS points with a
    validity mask. The point-level encoder (PointNetLite, see backbone)
    decides how to pool -- SNR-weighted attention -- rather than a fixed
    mean/std computed here.

    Returns a single tensor (not a tuple) so this drops into the existing
    generic train loop unchanged (x, y = batch; x.to(DEVICE)):

        out: [RADAR_LEN, RADAR_MAX_PTS, RADAR_POINT_FEATS + 1] float32
             out[..., :RADAR_POINT_FEATS] = point features
             out[..., -1]                 = validity mask (1.0/0.0)

    The backbone splits the mask channel back off internally.
    """
    files = sorted(glob.glob(os.path.join(tdir, "*.csv")))
    frames = []  # list of per-raw-frame point arrays [N_i, 6]
    for p in files:
        df = read_csv_any(p)
        if df is None:
            continue
        need = ["frame", "x", "y", "z", "v", "snr", "noise"]
        if not all(c in df.columns for c in need):
            continue
        for _, g in df.groupby("frame"):
            xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float32)
            v = g["v"].to_numpy(dtype=np.float32)[:, None]
            snr = (g["snr"].to_numpy(dtype=np.float32) / SNR_UNIT)[:, None]
            noise = (g["noise"].to_numpy(dtype=np.float32) / NOISE_UNIT)[:, None]
            frames.append(np.concatenate([xyz, v, snr, noise], axis=1))

    if len(frames) < 2:
        return torch.zeros(RADAR_LEN, RADAR_MAX_PTS, RADAR_POINT_FEATS + 1)

    # bin raw frames into RADAR_LEN contiguous groups, concatenate points
    # within each group -- this is the point-cloud analogue of the old
    # np.interp resample, but merges rather than discards
    n = len(frames)
    edges = np.linspace(0, n, RADAR_LEN + 1).astype(int)
    pts = np.zeros((RADAR_LEN, RADAR_MAX_PTS, RADAR_POINT_FEATS), dtype=np.float32)
    mask = np.zeros((RADAR_LEN, RADAR_MAX_PTS), dtype=bool)

    for i in range(RADAR_LEN):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        bin_pts = frames[lo:hi]
        if not bin_pts:
            continue
        merged = np.concatenate(bin_pts, axis=0)  # [M, 6]

        if train and AUG_POINT_DROPOUT > 0 and len(merged) > 1:
            keep = np.random.rand(len(merged)) > AUG_POINT_DROPOUT
            if keep.any():
                merged = merged[keep]

        if len(merged) > RADAR_MAX_PTS:
            # keep the strongest-SNR points rather than an arbitrary slice
            order = np.argsort(-merged[:, 4])  # snr_norm column
            merged = merged[order[:RADAR_MAX_PTS]]

        k = len(merged)
        if train and AUG_POINT_JITTER > 0:
            merged = merged.copy()
            merged[:, :3] += np.random.normal(0, AUG_POINT_JITTER, (k, 3)).astype(np.float32)

        pts[i, :k] = merged
        mask[i, :k] = True

    out = np.concatenate([pts, mask.astype(np.float32)[..., None]], axis=-1)
    return torch.from_numpy(out)