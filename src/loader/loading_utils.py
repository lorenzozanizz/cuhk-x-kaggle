from .dataset_config import *

import os

import numpy as np
import pandas as pd
import torch

# big libraries
from PIL import Image


def read_csv_any(path):
    for enc in ("utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except IOError:
            continue
    return None

def list_frames(d):
    img_ext = (".png", ".jpg", ".jpeg")
    fs = [f for f in sorted(os.listdir(d)) if f.lower().endswith(img_ext)]
    return [os.path.join(d, f) for f in fs]


def sample_indices(n, k, jitter):
    """ Uniform frame index sampling in relative clip time, with small random
    jitter as temporal augmentation. """
    idx = np.linspace(0, n - 1, k)
    if jitter and n > k:
        idx = idx + np.random.uniform(-0.5, 0.5, size=k) * (n / k)
    return np.clip(np.round(idx), 0, n - 1).astype(int)


def load_vision(tdir, train, channels):
    """ Frames scaled to [0, 1]. Spatial augmentation parameters are drawn once
    per clip so all frames receive the same crop, flip and gain. """
    files = list_frames(tdir)
    if not files:
        return torch.zeros(VISION_FRAMES, channels, VISION_SIZE, VISION_SIZE)
    idx = sample_indices(len(files), VISION_FRAMES, jitter=train)
    load_size = int(VISION_SIZE * AUG_CROP_FACTOR)
    max_off = load_size - VISION_SIZE
    if train:
        ox = np.random.randint(0, max_off + 1)
        oy = np.random.randint(0, max_off + 1)
        flip = np.random.rand() < 0.5
        gain = np.random.uniform(*AUG_GAIN_RANGE)
    else:
        ox = oy = max_off // 2
        flip = False
        gain = 1.0
    frames = []
    for i in idx:
        img = Image.open(files[i]).convert("RGB" if channels == 3 else "L")
        img = img.resize((load_size, load_size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        if a.ndim == 2:
            a = a[:, :, None]
        a = a[oy:oy + VISION_SIZE, ox:ox + VISION_SIZE]
        if flip:
            a = a[:, ::-1]
        frames.append(np.clip(a * gain, 0.0, 1.0))
    x = np.stack(frames).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(x))

