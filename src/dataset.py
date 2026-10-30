"""Dataset for stage-C image metadata. No split creation or augmentation."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ChartDataset(Dataset):
    """Return float32 RGB [3,224,224] in [0,1] and integer label 0 or 1.

    image_path entries are relative to project_root. Apply ImageNet
    normalization separately when using ImageNet-pretrained weights.
    """
    def __init__(self, metadata_csv, split, project_root=PROJECT_ROOT):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.root = Path(project_root).resolve()
        frame = pd.read_csv(metadata_csv)
        if not {"split", "image_path", "label"}.issubset(frame.columns):
            raise ValueError("metadata must include split, image_path, and label")
        self.rows = frame.loc[frame["split"] == split, ["image_path", "label"]].reset_index(drop=True)
        if self.rows.empty or not self.rows["label"].isin([0, 1]).all():
            raise ValueError("split is empty or contains invalid labels")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        path = (self.root / str(row["image_path"])).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("image path must stay within project root")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                raise ValueError(f"expected RGB 224x224 image: {path}")
            pixels = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(255.0)
        return tensor, int(row["label"])
