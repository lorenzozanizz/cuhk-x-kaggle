"""
DatasetExplorer for the CUHK-X (Small Model Track) dataset.

Expected layout (after extracting the training zip):
    root/
        HAR/data/<modality>/<action>/<user>/<trial>/<files>

Modalities: Depth_Color, IR, Thermal, IMU, Radar, Skeleton
Image-based modalities are counted by number of frame files.
CSV-based modalities (IMU, Radar) are counted by number of rows.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

IMAGE_MODALITIES = {"Depth_Color", "IR", "Thermal"}
CSV_MODALITIES = {"IMU", "Radar"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class DatasetExplorer:
    """ Scans the CUHK-X training tree and reports basic dataset statistics. """

    def __init__(self, data_root, class_mapping_csv=None):
        self.data_root = data_root
        self.class_mapping = self._load_class_mapping(class_mapping_csv)
        self.records = []  # one entry per (modality, action, user, trial)
        self.modalities = set()
        self.actions = set()

    def _load_class_mapping(self, path) -> dict:
        """

        :param path:
        :return:
        """
        if path is None or not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8-sig") as f:
            return {
                row["action_id"]: row["action_name"] for row in csv.DictReader(f)
            }

    def _find_trials(self):
        """Collect (modality, action, user, trial, trial_path) tuples to process."""
        trials = []
        for modality in sorted(os.listdir(self.data_root)):
            modality_path = os.path.join(self.data_root, modality)
            if not os.path.isdir(modality_path):
                continue
            for action in sorted(os.listdir(modality_path)):
                action_path = os.path.join(modality_path, action)
                if not os.path.isdir(action_path):
                    continue
                for user in sorted(os.listdir(action_path)):
                    user_path = os.path.join(action_path, user)
                    if not os.path.isdir(user_path):
                        continue
                    for trial in sorted(os.listdir(user_path)):
                        trial_path = os.path.join(user_path, trial)
                        if os.path.isdir(trial_path):
                            trials.append((modality, action, user, trial, trial_path))
        return trials

    def scan(self):
        """Walk the directory tree and build the internal record list."""
        self.records = []
        trials = self._find_trials()

        for modality, action, user, trial, trial_path in tqdm(trials, desc="Scanning clips"):
            length = self._sequence_length(modality, trial_path)
            self.records.append({
                "modality": modality,
                "action": action,
                "user": user,
                "trial": trial,
                "length": length,
            })
            self.modalities.add(modality)
            self.actions.add(action)

    def _sequence_length(self, modality, trial_path):
        """Return a length value describing the size of one clip's modality data."""
        files = [f for f in os.listdir(trial_path)
                 if os.path.isfile(os.path.join(trial_path, f))]

        if modality in IMAGE_MODALITIES:
            return sum(1 for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)

        if modality in CSV_MODALITIES:
            total_rows = 0
            for f in files:
                if f.lower().endswith(".csv"):
                    total_rows += self._count_csv_rows(os.path.join(trial_path, f))
            return total_rows

        # Skeleton and any other modality: fall back to counting all files.
        return len(files)

    def _count_csv_rows(self, csv_path):
        try:
            with open(csv_path, encoding="utf-8-sig") as f:
                row_count = sum(1 for _ in csv.reader(f))
            return max(row_count - 1, 0)  # exclude header
        except OSError:
            return 0

    def _samples(self):
        """Return the set of unique (action, user, trial) clips."""
        return {(r["action"], r["user"], r["trial"]) for r in self.records}

    def total_samples(self):
        return len(self._samples())

    def modality_occurrence_table(self):
        """Return {action: {modality: count}} of clips containing each modality."""
        table = defaultdict(lambda: defaultdict(int))
        for r in self.records:
            table[r["action"]][r["modality"]] += 1
        return table

    def average_sequence_length(self):
        """Return {modality: average_length} across all recorded clips."""
        lengths = defaultdict(list)
        for r in self.records:
            lengths[r["modality"]].append(r["length"])
        return {m: float(np.mean(v)) if v else 0.0 for m, v in lengths.items()}

    def missing_modality_report(self):
        """Return {modality: number_of_samples_missing_it}."""
        samples = self._samples()
        present = defaultdict(set)
        for r in self.records:
            present[r["modality"]].add((r["action"], r["user"], r["trial"]))

        missing = {}
        for modality in self.modalities:
            missing[modality] = len(samples - present[modality])
        return missing

    def samples_per_class(self):
        """Return {action: number_of_unique_samples} for that class."""
        counts = defaultdict(set)
        for r in self.records:
            counts[r["action"]].add((r["user"], r["trial"]))
        return {
            action: len(trials) for action, trials in counts.items()
        }

    def print_report(self):
        print(f"Total samples (unique action/user/trial clips): {self.total_samples()}")
        print(f"Actions found: {len(self.actions)}")
        print(f"Modalities found: {sorted(self.modalities)}")

        print("\nAverage sequence length per modality:")
        for modality, avg in sorted(self.average_sequence_length().items()):
            print(f"  {modality:<12} avg length = {avg:.1f}")

        print("\nMissing modality counts (samples lacking that modality):")
        total = self.total_samples()
        for modality, count in sorted(self.missing_modality_report().items()):
            print(f"  {modality:<12} missing in {count}/{total} samples")

        print("\nSamples per class:")
        for action, count in sorted(self.samples_per_class().items()):
            print(f"  {action:<20} {count} samples")

    def plot_samples_per_class(self, save_path=None):
        counts = self.samples_per_class()
        actions = sorted(counts.keys())
        values = [counts[a] for a in actions]

        fig, ax = plt.subplots(figsize=(max(8, len(actions) * 0.3), 5))
        ax.bar(actions, values, color="darkslateblue")
        ax.set_title("Samples per class")
        ax.set_xlabel("Action (class)")
        ax.set_ylabel("Number of samples")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_modality_occurrence(self, save_path=None):
        table = self.modality_occurrence_table()
        actions = sorted(table.keys())
        modalities = sorted(self.modalities)

        matrix = np.array([[table[a].get(m, 0) for m in modalities] for a in actions])

        fig, ax = plt.subplots(figsize=(max(6, len(modalities) * 1.2),
                                         max(6, len(actions) * 0.3)))
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")

        ax.set_xticks(range(len(modalities)))
        ax.set_xticklabels(modalities, rotation=45, ha="right")
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, fontsize=7)

        # Gridlines between cells so the matrix reads as a grid, not solid bands.
        ax.set_xticks(np.arange(-0.5, len(modalities), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(actions), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)

        # Annotate each cell with its count so exact values are readable.
        for i in range(len(actions)):
            for j in range(len(modalities)):
                value = matrix[i, j]
                text_color = "white" if value > matrix.max() / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center",
                        fontsize=6, color=text_color)

        ax.set_title("Modality occurrence per action")
        ax.set_xlabel("Modality")
        ax.set_ylabel("Action")
        fig.colorbar(im, ax=ax, label="Number of clips")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_average_sequence_length(self, save_path=None):
        avg_lengths = self.average_sequence_length()
        modalities = sorted(avg_lengths.keys())
        values = [avg_lengths[m] for m in modalities]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(modalities, values, color="steelblue")
        ax.set_title("Average sequence length per modality")
        ax.set_xlabel("Modality")
        ax.set_ylabel("Average length (frames or rows)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_missing_modalities(self, save_path=None):
        missing = self.missing_modality_report()
        modalities = sorted(missing.keys())
        values = [missing[m] for m in modalities]
        total = self.total_samples()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(modalities, values, color="indianred")
        ax.axhline(total, color="black", linestyle="--", linewidth=1,
                   label=f"Total samples ({total})")
        ax.set_title("Samples missing each modality")
        ax.set_xlabel("Modality")
        ax.set_ylabel("Number of samples missing")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

if __name__ == "__main__":

    TRAIN_ROOT = "C:/Users/picul/Documents/Research/Kaggle/UHK-X/Training/"
    data_root = TRAIN_ROOT + "data/HAR/data"
    class_mapping_csv = TRAIN_ROOT+ "class_mapping.csv"

    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        print("Extract the training zip first, then update data_root if needed.")
    else:

        explorer = DatasetExplorer(data_root, class_mapping_csv)
        explorer.scan()
        explorer.print_report()

        explorer.plot_modality_occurrence(save_path="modality_occurrence.png")
        explorer.plot_average_sequence_length(save_path="avg_sequence_length.png")
        explorer.plot_missing_modalities(save_path="missing_modalities.png")
        explorer.plot_samples_per_class(save_path="samples_per_class.png")

        plt.show()