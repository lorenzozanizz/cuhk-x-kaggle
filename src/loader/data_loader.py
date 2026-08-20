from .dataset_config import *

from .imu_loader import load_imu
from .ir_loader import load_ir
from .skeleton_loader import load_skeleton
from .thermal_loader import load_thermal
from .depth_loader import load_depth
from .radar_loader import load_radar

from typing import Union, Optional, Sized, Callable
from pathlib import Path
import os
import random
import tqdm
import torch
import collections

# big libraries
from torch.utils.data import Dataset

import enum

class Modality(enum.Enum):
    DEPTH_COLOR = "Depth_Color"
    IMU = "IMU"
    IR = "IR"
    RADAR = "Radar"
    SKELETON = "Skeleton"
    THERMAL = "Thermal"

class DataIndex:

    def __init__(self, data_root: Union[str, Path], cache_root: Union[str, Path]):
        self.data_root = data_root
        self.cache_root = cache_root
        self._samples = {}
        self._build()

    def _build(self) -> None:
        """ Scan HAR/data/<modality>/<action>/<user>/<trial> and merge the modality
        folders of the same (action, user, trial) into one sample record. """

        # use tqdm to show the progress bar
        for modality in tqdm.tqdm(MODALITIES):

            data_root = os.path.join(self.data_root, modality)
            if not os.path.isdir(data_root):
                continue
            for action in sorted(os.listdir(data_root)):
                adir = os.path.join(data_root, action)
                if not os.path.isdir(adir):
                    continue
                try:
                    action_id = int(action.split("_")[0])
                except ValueError:
                    continue
                for user in sorted(os.listdir(adir)):
                    usr_dir = os.path.join(adir, user)
                    if not os.path.isdir(usr_dir):
                        continue
                    for trial in sorted(os.listdir(usr_dir)):
                        tdir = os.path.join(usr_dir, trial)
                        if not os.path.isdir(tdir):
                            continue
                        key = (action_id, user, trial)
                        entry = self._samples.setdefault(
                            key, {"label": action_id, "user": user, "paths": {}})
                        entry["paths"][modality] = tdir

    def get_samples(self) -> Sized:
        return self._samples.values()

    def statistics(self) -> None:
        """ Print a summary of the indexed dataset: counts per modality,
        per action label, per user, and how many samples have complete
        vs. missing modality coverage. """

        total_samples = len(self._samples)
        print(f"{'=' * 60}")
        print(f"DataIndex statistics")
        print(f"{'=' * 60}")
        print(f"Total unique (action, user, trial) samples: {total_samples}")

        if total_samples == 0:
            return

        modality_counts = collections.Counter()
        for entry in self._samples.values():
            for modality in entry["paths"]:
                modality_counts[modality] += 1

        print(f"\nPer-modality sample counts (out of {total_samples}):")
        for modality in MODALITIES:
            count = modality_counts.get(modality, 0)
            pct = 100.0 * count / total_samples
            print(f"  {modality:<15s}: {count:>6d}  ({pct:5.1f}%)")

        n_modalities = len(MODALITIES)
        complete = sum(
            1 for entry in self._samples.values()
            if len(entry["paths"]) == n_modalities
        )
        incomplete = total_samples - complete
        print(f"\nSamples with all {n_modalities} modalities: {complete}")
        print(f"Samples with missing modalities:      {incomplete}")

        if incomplete:
            missing_breakdown = collections.Counter()
            for entry in self._samples.values():
                missing = tuple(sorted(set(MODALITIES) - set(entry["paths"])))
                if missing:
                    missing_breakdown[missing] += 1
            print("\nBreakdown of missing-modality combinations:")
            for missing, cnt in sorted(missing_breakdown.items(), key=lambda x: -x[1]):
                print(f"  missing {missing}: {cnt} samples")

        action_counts = collections.Counter(entry["label"] for entry in self._samples.values())
        print(f"\nActions found: {len(action_counts)}")
        for action_id in sorted(action_counts):
            print(f"  action {action_id:>3d}: {action_counts[action_id]:>6d} samples")

        user_counts = collections.Counter(entry["user"] for entry in self._samples.values())
        print(f"\nUsers found: {len(user_counts)}")
        for user in sorted(user_counts):
            print(f"  {user:<15s}: {user_counts[user]:>6d} samples")

        print(f"{'=' * 60}\n")

    def get_samples_by_modality(self, modality: Modality) -> list:
        """ Return only the samples that have data for the given modality. """
        if not isinstance(modality, Modality):
            raise TypeError(f"modality must be a Modality enum member, got {type(modality)!r}")

        mod_str = modality.value
        return [
            entry for entry in self._samples.values()
            if mod_str in entry["paths"]
        ]

def split_by_user(samples, val_fraction=0.2, seed=0):
    """ Group split: validation users never appear in training. This matches the
    cross subject test condition. A random clip split leaks subject identity
    and overestimates accuracy."""
    users = sorted({s["user"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(users)
    n_val = max(1, int(round(len(users) * val_fraction)))
    val_users = set(users[:n_val])
    train = [s for s in samples if s["user"] not in val_users]
    val = [s for s in samples if s["user"] in val_users]
    return train, val, sorted(val_users)


def split_random(samples, val_fraction=0.2, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    cut = int(len(samples) * (1.0 - val_fraction))
    return [samples[i] for i in idx[:cut]], [samples[i] for i in idx[cut:]]


# Again, a fully static class for namespace cleanness
class Loaders:


    @staticmethod
    def get(mode) -> Optional[Callable]:
        _loaders = {
            "Depth_Color": load_depth,
            "IR":  load_ir,
            "Thermal": load_thermal,
            "IMU": load_imu,
            "Radar": load_radar,
            "Skeleton": load_skeleton,
        }
        return _loaders.get(mode, None)


class SingleModalityDataset(Dataset):
    """ Used for individual backbone training. Only samples that actually have
    the requested modality are indexed. """

    def __init__(self, samples, modality, train):
        self.items = [s for s in samples if modality in s["paths"]]
        self.modality = modality
        self.train = train
        self.loader = Loaders.get(modality)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        x = self.loader(s["paths"][self.modality], self.train)
        return x, s["label"]


class MultiModalityDataset(Dataset):
    """Returns a dict of fixed shape tensors plus a presence mask. Missing
    modalities are zero tensors with mask 0, so the default collate works."""

    def __init__(self, samples, modalities, train):
        self.modalities = list(modalities)
        self.items = [s for s in samples
                      if any(m in s["paths"] for m in self.modalities)]
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        out, mask = {}, torch.zeros(len(self.modalities))
        for j, m in enumerate(self.modalities):
            if m in s["paths"]:
                out[m] = Loaders.get(m)(s["paths"][m], self.train)
                mask[j] = 1.0
            else:
                out[m] = torch.zeros(SHAPES[m])
        return out, mask, s["label"]
