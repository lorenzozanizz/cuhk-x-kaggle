import numpy as np
import torch
from PIL import Image

from .data_loader import *
from .loading_utils import *

# Some constants for unifying image sequences
VISION_FRAMES = 16
VISION_SIZE = 112
AUG_CROP_FACTOR = 1.15
AUG_GAIN_RANGE = (0.85, 1.15)
AUG_ROT_DEG = 8.0
AUG_CUTOUT_PROB = 0.5
AUG_CUTOUT_NUM = (1, 3)
AUG_CUTOUT_FRAC = (0.05, 0.20)

MODALITY_STATS = {
    "depth":   dict(channels=3, mean=(0.35, 0.30, 0.30), std=(0.25, 0.22, 0.22)),
    "ir":      dict(channels=1, mean=(0.45,),            std=(0.22,)),
    "thermal": dict(channels=3, mean=(0.40, 0.32, 0.32), std=(0.25, 0.22, 0.22)),
}

def load_vision(tdir, train, modality):
    """ Returns (x, present) where x is [VISION_FRAMES, C, VISION_SIZE, VISION_SIZE]
    normalized to zero-mean/unit-std, and present is a bool
    telling the joint model whether this clip actually had this modality.
    Augmentation parameters (crop, flip, gain, rotation, cutout boxes) are
    drawn once per clip so every frame in the clip receives the same
    transform
    """
    stats = MODALITY_STATS[modality]
    channels = stats["channels"]
    files = list_frames(tdir)

    if not files:
        x = torch.zeros(VISION_FRAMES, channels, VISION_SIZE, VISION_SIZE)
        return x, False

    idx = sample_indices(len(files), VISION_FRAMES, jitter=train)
    load_size = int(VISION_SIZE * AUG_CROP_FACTOR)
    max_off = load_size - VISION_SIZE

    if train:
        ox = np.random.randint(0, max_off + 1)
        oy = np.random.randint(0, max_off + 1)
        flip = np.random.rand() < 0.5
        gain = np.random.uniform(*AUG_GAIN_RANGE)
        angle = np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG)
        n_cut = (np.random.randint(AUG_CUTOUT_NUM[0], AUG_CUTOUT_NUM[1] + 1)
                 if np.random.rand() < AUG_CUTOUT_PROB else 0)
    else:
        ox = oy = max_off // 2
        flip, gain, angle, n_cut = False, 1.0, 0.0, 0

    cutout_boxes = []
    for _ in range(n_cut):
        frac = np.random.uniform(*AUG_CUTOUT_FRAC)
        ch = cw = int(VISION_SIZE * frac)
        cy = np.random.randint(0, VISION_SIZE - ch + 1)
        cx = np.random.randint(0, VISION_SIZE - cw + 1)
        cutout_boxes.append((cy, cx, ch, cw))

    fill = 0 if channels == 1 else (0, 0, 0)  # black padding for rotation
    frames = []

    # RANDOM AUGMENTATION FOR INPUT IMAGES
    for i in idx:
        img = Image.open(files[i]).convert("RGB" if channels == 3 else "L")
        img = img.resize((load_size, load_size), Image.BILINEAR)
        if angle != 0.0:
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill)
        a = np.asarray(img, dtype=np.float32) / 255.0
        if a.ndim == 2:
            a = a[:, :, None]
        a = a[oy:oy + VISION_SIZE, ox:ox + VISION_SIZE]
        if flip:
            a = a[:, ::-1]
        a = np.clip(a * gain, 0.0, 1.0)
        for (cy, cx, ch, cw) in cutout_boxes:
            a[cy:cy + ch, cx:cx + cw] = 0.0  # black = "invalid/unknown", matches
        # depth's own black-for-unregistered
        # convention, so it's semantically
        # consistent to reuse it for masking
        frames.append(a)

    x = np.stack(frames).transpose(0, 3, 1, 2)
    x = torch.from_numpy(np.ascontiguousarray(x))
    mean = torch.tensor(stats["mean"]).view(1, channels, 1, 1)
    std = torch.tensor(stats["std"]).view(1, channels, 1, 1)
    x = (x - mean) / std
    return x, True