"""
GreenTea-8 Dataset
===================
PyTorch Dataset wrapper for the tea cultivar classification dataset.

The dataset on disk uses torchvision.datasets.ImageFolder with 8 native class
subdirectories (matching the actual server dataset at /data1/aresnet_project/dataset/).

Two modes are supported:
1. Native 8-class mode: loads all 8 class folders as-is
2. Merge mode: merges Xinyang_Maojian + Zhuyeqing into one category (7 classes),
   matching the paper's GreenTea-8 definition

Dataset Statistics (Server 243 clean version):
-----------------------------------------------
Native 8 classes (1707T / 296V):
  Anji_Baicha:      248T / 50V
  Biluochun:        211T / 32V
  Lishui_Xiangcha:  202T / 45V
  Liu_an_Guapian:   174T / 30V
  Longjing:         294T / 36V
  Taiping_Houkui:   219T / 40V
  Xinyang_Maojian:  225T / 35V
  Zhuyeqing:        134T / 28V

Merged 7 classes (paper version):
  XYMJ_ZYQ = Xinyang_Maojian + Zhuyeqing merged: 359T / 63V

Directory Structure:
--------------------
dataset/data/
├── train/
│   ├── Anji_Baicha/    (248 images)
│   ├── Biluochun/      (211 images)
│   ├── ...
│   └── Zhuyeqing/      (134 images)
└── val/
    └── (same class subdirectories)

Reference:
----------
CPC-Net: Learning Prototype-Conditioned Convolutional Kernels
for Fine-Grained Tea Cultivar Classification with Limited Data (2026)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image


# ============================================================================
# Native 8-class labels (matching server dataset directories)
# ============================================================================
NATIVE_8_CLASSES = [
    'Anji_Baicha',
    'Biluochun',
    'Lishui_Xiangcha',
    'Liu_an_Guapian',
    'Longjing',
    'Taiping_Houkui',
    'Xinyang_Maojian',
    'Zhuyeqing',
]

# Human-readable English names
CLASS_EN = {
    'Anji_Baicha': 'Anji Baicha',
    'Biluochun': 'Biluochun',
    'Lishui_Xiangcha': 'Lishui Xiangcha',
    'Liu_an_Guapian': "Liu'an Guapian",
    'Longjing': 'Longjing',
    'Taiping_Houkui': 'Taiping Houkui',
    'Xinyang_Maojian': 'Xinyang Maojian',
    'Zhuyeqing': 'Zhuyeqing',
}

# Human-readable Chinese names
CLASS_CN = {
    'Anji_Baicha': '安吉白茶',
    'Biluochun': '碧螺春',
    'Lishui_Xiangcha': '溧水香茶',
    'Liu_an_Guapian': '六安瓜片',
    'Longjing': '龙井',
    'Taiping_Houkui': '太平猴魁',
    'Xinyang_Maojian': '信阳毛尖',
    'Zhuyeqing': '竹叶青',
}

# Merged 7-class mode (paper's GreenTea-8, which is actually 7 classes)
MERGED_7_CLASSES = [
    'Anji_Baicha',
    'Biluochun',
    'Lishui_Xiangcha',
    'Liu_an_Guapian',
    'Longjing',
    'Taiping_Houkui',
    'XYMJ_ZYQ',  # Xinyang_Maojian + Zhuyeqing merged
]

NUM_CLASSES_NATIVE = len(NATIVE_8_CLASSES)  # 8
NUM_CLASSES_MERGED = len(MERGED_7_CLASSES)  # 7


class MergeGreenTeaDataset(Dataset):
    """Dataset wrapper that merges Xinyang_Maojian and Zhuyeqing into one class.

    Wraps torchvision ImageFolder. When merge_xyzy=True, samples from
    Xinyang_Maojian (label 6) and Zhuyeqing (label 7) are both assigned
    label 6 (XYMJ_ZYQ), producing a 7-class dataset matching the paper.

    Args:
        root_dir: Path to train/ or val/ directory with class subdirectories.
        transform: Image transforms.
        merge_xyzy: If True, merge Xinyang_Maojian+Zhuyeqing into one class.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[object] = None,
        merge_xyzy: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.merge_xyzy = merge_xyzy

        # ImageFolder auto-discovers class directories alphabetically
        # The server dataset order: Anji_Baicha, Biluochun, Lishui_Xiangcha,
        # Liu_an_Guapian, Longjing, Taiping_Houkui, Xinyang_Maojian, Zhuyeqing
        self.base_dataset = ImageFolder(str(self.root_dir))

        if merge_xyzy:
            self.num_classes = NUM_CLASSES_MERGED
            self.classes = MERGED_7_CLASSES
            # Remap: 0-5 stay, 6(XYMJ) and 7(ZYQ) both → 6(XYMJ_ZYQ)
            self._remap = {i: i for i in range(6)}
            self._remap[6] = 6  # Xinyang_Maojian → XYMJ_ZYQ
            self._remap[7] = 6  # Zhuyeqing → XYMJ_ZYQ
        else:
            self.num_classes = NUM_CLASSES_NATIVE
            self.classes = NATIVE_8_CLASSES
            self._remap = None

        # Store samples with remapped labels
        self.samples = []
        for path, label in self.base_dataset.samples:
            final_label = self._remap[label] if self._remap else label
            self.samples.append((path, final_label))

        # Cache class_to_idx
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_counts(self) -> Dict[str, int]:
        """Count samples per class (after any merge)."""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts


def create_dataloaders(
    data_dir: Union[str, Path],
    train_transform: object,
    val_transform: object,
    batch_size: int = 32,
    num_workers: int = 4,
    merge_xyzy: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders.

    Args:
        data_dir: Root directory containing 'train/' and 'val/' subdirectories.
        train_transform: Training transforms.
        val_transform: Validation transforms.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        merge_xyzy: If True, merge Xinyang_Maojian+Zhuyeqing into 7 classes.

    Returns:
        (train_loader, val_loader) tuple.
    """
    data_dir = Path(data_dir)

    train_ds = MergeGreenTeaDataset(
        root_dir=data_dir / 'train',
        transform=train_transform,
        merge_xyzy=merge_xyzy,
    )
    val_ds = MergeGreenTeaDataset(
        root_dir=data_dir / 'val',
        transform=val_transform,
        merge_xyzy=merge_xyzy,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================================
# Quick test
# ============================================================================
if __name__ == '__main__':
    from dataset.transforms import get_train_transforms, get_val_transforms
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'dataset/data'
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()

    print(f"Data dir: {data_dir}")
    print(f"Train dir exists: {os.path.isdir(os.path.join(data_dir, 'train'))}")

    if not os.path.isdir(os.path.join(data_dir, 'train')):
        print("Dataset not found. Download from server first.")
        sys.exit(1)

    for merge in [False, True]:
        train_ld, val_ld = create_dataloaders(data_dir, train_tf, val_tf, merge_xyzy=merge)
        print(f"\n{'='*50}")
        print(f"Merge XYMJ+ZYQ: {merge} | Classes: {train_ld.dataset.num_classes}")
        print(f"Train: {len(train_ld.dataset)}, Val: {len(val_ld.dataset)}")
        counts = train_ld.dataset.get_class_counts()
        for cls, cnt in counts.items():
            print(f"  {cls}: {cnt}")
