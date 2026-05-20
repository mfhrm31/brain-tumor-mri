"""
Brain tumor MRI dataset loader.

Wraps the combined Kaggle brain tumor MRI dataset structure:
    data_dir/
        Training/
            glioma/ ... meningioma/ ... pituitary/ ... notumor/ ...
        Testing/
            glioma/ ... meningioma/ ... pituitary/ ... notumor/ ...
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple


CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}


def get_default_transforms(
    image_size: int = 224,
    is_training: bool = True,
) -> transforms.Compose:
    """
    Build standard preprocessing transforms for brain MRI.

    Note: horizontal flip is NOT applied — brain asymmetry can be
    clinically relevant (e.g. tumor localization).

    Args:
        image_size: Target square image size
        is_training: Apply training augmentations if True

    Returns:
        torchvision Compose pipeline
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_training:
        ops = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    else:
        ops = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]

    return transforms.Compose(ops)


class BrainTumorMRIDataset(Dataset):
    """
    PyTorch Dataset for the brain tumor MRI dataset.

    Args:
        data_dir: Root directory containing Training/Testing splits
        split: One of 'train', 'test'
        transform: Optional torchvision transforms
        image_extensions: File extensions to scan for
    """

    SPLIT_FOLDERS = {
        "train": "Training",
        "test": "Testing",
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        if split not in self.SPLIT_FOLDERS:
            raise ValueError(
                f"split must be one of {list(self.SPLIT_FOLDERS.keys())}"
            )

        self.data_dir = Path(data_dir) / self.SPLIT_FOLDERS[split]
        self.split = split
        self.transform = transform
        self.image_extensions = image_extensions

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.data_dir}"
            )

        self.samples: List[Tuple[Path, int]] = []
        self._scan_files()

    def _scan_files(self):
        """Scan class subdirectories for image files."""
        for class_name in CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = CLASS_TO_IDX[class_name]
            for ext in self.image_extensions:
                for filepath in class_dir.glob(f"*{ext}"):
                    self.samples.append((filepath, class_idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.data_dir}. "
                f"Expected subdirectories: {CLASS_NAMES}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("L")  # Grayscale MRI

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def class_counts(self) -> dict:
        """Return number of samples per class."""
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts


def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders.

    Splits the official Training set into train/val using val_split.
    Uses the official Testing set as the held-out test set.

    Args:
        data_dir: Root directory of dataset
        image_size: Target image size
        batch_size: Batch size
        num_workers: DataLoader worker processes
        val_split: Fraction of training data for validation
        seed: Random seed for reproducible split

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    full_train = BrainTumorMRIDataset(
        data_dir, split="train",
        transform=get_default_transforms(image_size, is_training=True),
    )

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    # Use eval transforms for validation by wrapping the underlying dataset
    val_ds.dataset.transform = get_default_transforms(image_size, is_training=False)

    test_ds = BrainTumorMRIDataset(
        data_dir, split="test",
        transform=get_default_transforms(image_size, is_training=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Brain Tumor MRI Dataset utilities")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Class indices: {CLASS_TO_IDX}")
    print("\nFunctions available:")
    print("  - get_default_transforms(image_size, is_training)")
    print("  - BrainTumorMRIDataset(data_dir, split, transform)")
    print("  - get_dataloaders(data_dir, image_size, batch_size, val_split)")
