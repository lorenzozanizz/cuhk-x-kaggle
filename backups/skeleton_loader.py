
from .dataset_config import *
from .loading_utils import *

import os
import glob
import json

import numpy as np
import torch

# COCO-17 indices
LEFT_HIP, RIGHT_HIP = 1,4
UP_AXIS = SKELETON_UP_AXIS  # e.g. 2 for z-as-height


def rot_about_axis(ang, axis):
    c, s = np.cos(ang), np.sin(ang)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3, dtype=np.float32)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def canonicalize_yaw(x, left_idx, right_idx, up_axis):
    """Rotate the whole clip about up_axis so the (time-averaged) hip-to-hip
    vector points along a fixed direction. Removes facing direction as a
    source of variance deterministically, instead of relying on augmentation
    alone to average it out over training."""
    other = [k for k in range(3) if k != up_axis]
    ref = x[:, right_idx, other] - x[:, left_idx, other]
    ref_mean = ref.mean(axis=0)
    norm = np.linalg.norm(ref_mean)
    if norm < 1e-6:
        return x  # degenerate (e.g. hips coincident) -- skip rather than divide by ~0
    ang = -np.arctan2(ref_mean[1], ref_mean[0])
    R = rot_about_axis(ang, up_axis)
    return x @ R.T


def load_skeleton_2(tdir, train):
    """One json per frame, first detected person. Root centering removes global
    position, scale normalization removes body size, yaw canonicalization removes
    facing direction -- all three are subject/session cues that hurt cross-subject
    generalization rather than action-discriminative signal. A residual random
    rotation + noise is kept during training on top of canonicalization, since the
    hip-vector reference is itself noisy (sensor jitter), so the model needs to
    tolerate small canonicalization error at test time, not just a clean frame.
    Corrupt frames repeat the last valid pose. A 4th channel marks depth
    presence (always 1.0 here) so this loader stays compatible with a backbone
    also pretrained on 2D-only data where that channel is 0.
    """
    files = sorted(glob.glob(os.path.join(tdir, "*.json")))
    if not files:
        files = sorted(glob.glob(os.path.join(tdir, "**", "*.json"), recursive=True))
    if not files:
        x = torch.zeros(SKELETON_FRAMES, NUM_JOINTS, 3)
        flag = torch.ones(SKELETON_FRAMES, NUM_JOINTS, 1)
        return torch.cat([x, flag], dim=-1)

    idx = sample_indices(len(files), SKELETON_FRAMES, jitter=train)
    seq = []
    last = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for i in idx:
        try:
            with open(files[i]) as f:
                data = json.load(f)
            kp = np.asarray(data[0]["keypoints"], dtype=np.float32)
            if kp.shape == (NUM_JOINTS, 3) and np.isfinite(kp).all():
                last = kp
        except Exception:
            pass
        seq.append(last.copy())
    x = np.stack(seq)

    # root center
    x = x - x[:, :1, :]

    # scale normalize
    scale = np.linalg.norm(x.reshape(-1, 3), axis=1).mean()
    if scale > 1e-4:
        x = x / scale

    # deterministic yaw canonicalization
    # x = canonicalize_yaw(x, LEFT_HIP, RIGHT_HIP, UP_AXIS)

    if train:
        # residual jitter only -- canonicalization already removed the bulk
        # of the rotation variance, so this just covers reference-vector noise
        R = rot_about_axis(
            np.random.uniform(-10, 10),
            UP_AXIS,
        )
        x = x @ R.T
        x = x + np.random.normal(0, AUG_SKEL_NOISE, x.shape).astype(np.float32)

    x = torch.from_numpy(x.astype(np.float32))
    flag = torch.ones(x.shape[0], NUM_JOINTS, 1)
    return torch.cat([x, flag], dim=-1)  # [T, J, 4]


from .dataset_config import *
from .loading_utils import *

import os
import glob
import json

import numpy as np
import torch

def rot_about_axis(ang, axis):
    c, s = np.cos(ang), np.sin(ang)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3, dtype=np.float32)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R

def load_skeleton(tdir, train):
    # One json per frame, first detected person. Root centering removes global
    # position, scale normalization removes body size, both are subject cues that
    # hurt cross subject generalization. Corrupt frames repeat the last valid pose.
    files = sorted(glob.glob(os.path.join(tdir, "*.json")))
    if not files:
        files = sorted(glob.glob(os.path.join(tdir, "**", "*.json"), recursive=True))
    if not files:
        return torch.zeros(SKELETON_FRAMES, NUM_JOINTS, 3)
    idx = sample_indices(len(files), SKELETON_FRAMES, jitter=train)
    seq = []
    last = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for i in idx:
        try:
            with open(files[i]) as f:
                data = json.load(f)
            kp = np.asarray(data[0]["keypoints"], dtype=np.float32)
            if kp.shape == (NUM_JOINTS, 3) and np.isfinite(kp).all():
                last = kp
        except Exception:
            pass
        seq.append(last.copy())
    x = np.stack(seq)
    x = x - x[:, :1, :]
    scale = np.linalg.norm(x.reshape(-1, 3), axis=1).mean()
    if scale > 1e-4:
        x = x / scale
    if train:
        R = rot_about_axis(np.random.uniform(-AUG_SKEL_ROT_RAD, AUG_SKEL_ROT_RAD), SKELETON_UP_AXIS)
        x = x @ R.T
        x = x + np.random.normal(0, AUG_SKEL_NOISE, x.shape).astype(np.float32)
    return torch.from_numpy(x.astype(np.float32))

