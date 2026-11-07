"""Explicit RGB pairs. Geometry evidence never enters the model input."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data_utils import img_path_to_data


class GoalRelationDataset(Dataset):
    def __init__(self, rows, data_root, image_size):
        self.rows = rows
        self.root = Path(data_root)
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def image(self, relative_path):
        value = img_path_to_data(self.root / relative_path, self.image_size)
        if value.shape[0] != 3:
            raise ValueError(f"Expected RGB input: {relative_path}")
        return value

    def __getitem__(self, index):
        row = self.rows[index]
        # The manifest loader has already checked label semantics and paths.
        return {
            "obs_image": torch.cat([self.image(p) for p in row["context_images"]]),
            "goal_image": self.image(row["goal_image"]),
            "reachable": torch.tensor(row["reachable"] if row["reachable"] is not None else 0, dtype=torch.float32),
            "reach_valid": torch.tensor(row["reachable"] is not None),
            "distance_m": torch.tensor(row["path_length_m"] if row["distance_valid"] else 0.0, dtype=torch.float32),
            "distance_valid": torch.tensor(row["distance_valid"]),
            "row_index": index,
        }
