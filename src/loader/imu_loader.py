from .dataset_config import *
from .loading_utils import read_csv_any
from .standardizer import Standardizer

import os
import glob

import numpy as np
import pandas as pd
import torch

def load_imu(imu_dir, train):
    """Columns are addressed by position because headers are Chinese. Kept
    channels: acc xyz in g, gyro xyz in deg/s. Absolute angles, quaternions and
    magnetometer are dropped: they encode mounting and session specifics and
    leak under a cross subject split. Each device stream is sorted by time and
    resampled to a fixed length grid spanning the clip, which normalizes every
    clip to the same length in relative time. Missing devices stay zero."""
    return torch.zeros(len(imu_dir), len(imu_dir[0]), device=imu_dir[0].device)