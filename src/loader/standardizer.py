from .dataset_config import *

import os
import json
import numpy as np


class Standardizer:
    """ A standardizer class to standardize data across different modalities (mainly IMU
    and radar modalities). This should load from a cached value.
    """

    _STATS = None

    @staticmethod
    def load_stats(path=STATS_PATH):
        """ Get the statistics directly without the standardize interface """
        if Standardizer._STATS is None and os.path.isfile(path):
            with open(path) as f:
                Standardizer._STATS = json.load(f)
        return Standardizer._STATS

    @staticmethod
    def fit_stats(train_samples, path=STATS_PATH, limit_per_modality=800):
        from .data_loader import Loaders   # <-- lazy import, breaks the cycle

        acc = {
            "IMU": [np.zeros(IMU_CH), np.zeros(IMU_CH), np.zeros(IMU_CH)],
            "Radar": [np.zeros(RADAR_FEATURES), np.zeros(RADAR_FEATURES), np.zeros(RADAR_FEATURES)]
        }
        seen = {"IMU": 0, "Radar": 0}
        for s in train_samples:
            for m in ("IMU", "Radar"):
                if m not in s["paths"] or seen[m] >= limit_per_modality:
                    continue
                x = Loaders.get(m)(s["paths"][m], False).numpy()
                filled = x.any(axis=1) if m == "IMU" \
                    else np.ones(x.shape[0], dtype=bool)
                t = x.shape[1]
                acc[m][0][filled] += x[filled].sum(axis=1)
                acc[m][1][filled] += (x[filled] ** 2).sum(axis=1)
                acc[m][2][filled] += t
                seen[m] += 1

        stats = {}
        for m in ("IMU", "Radar"):
            n = np.maximum(acc[m][2], 1.0)
            mean = acc[m][0] / n
            var = np.maximum(acc[m][1] / n - mean ** 2, 1e-6)
            stats[m] = {"mean": mean.tolist(),
                        "std": np.sqrt(var).tolist(),
                        "clips": seen[m]}
        with open(path, "w") as f:
            json.dump(stats, f)
        Standardizer._STATS = stats
        return stats

    @staticmethod
    def standardize(x, key, row_mask=None):
        """ Load the corresponding data and use it to standardize the value """
        if Standardizer._STATS is None or key not in Standardizer._STATS:
            return x
        mean = np.asarray(Standardizer._STATS[key]["mean"], dtype=np.float32)
        std = np.asarray(Standardizer._STATS[key]["std"], dtype=np.float32)
        if row_mask is None:
            return (x - mean[:, None]) / std[:, None]
        x = x.copy()
        x[row_mask] = (x[row_mask] - mean[row_mask, None]) / std[row_mask, None]
        return x

    @staticmethod
    def ensure(stand_set):
        """ Ensure that the data is present in teh local cache """
        if Standardizer.load_stats() is None:
            Standardizer.fit_stats(stand_set)