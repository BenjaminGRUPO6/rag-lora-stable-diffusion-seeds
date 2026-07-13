from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

EXPECTED_CLASSES: tuple[str, ...] = (
    "intact",
    "spotted",
    "immature",
    "broken",
    "skin_damaged",
)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
SPLITS: tuple[str, ...] = ("train", "validation", "test")


class OrderedImageFolder(ImageFolder):
    """ImageFolder variant that preserves the expected soybean class order."""

    def __init__(
        self,
        root: str | Path,
        expected_classes: Sequence[str],
        transform: v2.Transform | None = None,
    ) -> None:
        self.expected_classes = tuple(expected_classes)
        super().__init__(root=str(root), transform=transform)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """Return classes in the configured order after validating directories."""
        root = Path(directory)
        found = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
        expected = list(self.expected_classes)
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        if missing or extra:
            raise ValueError(
                f"Invalid classes under {root}. Missing={missing or 'none'}, "
                f"extra={extra or 'none'}."
            )
        return expected, {class_name: index for index, class_name in enumerate(expected)}


@dataclass(frozen=True)
class VisionDatasets:
    """Container for soybean split datasets and validated class mapping."""

    train: Dataset
    validation: Dataset
    test: Dataset
    class_to_idx: dict[str, int]


@dataclass(frozen=True)
class VisionDataLoaders:
    """Container for soybean split dataloaders and class distributions."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    class_to_idx: dict[str, int]
    distributions: dict[str, dict[str, int]]


def build_transforms(image_size: int, train: bool) -> v2.Compose:
    """Build deterministic evaluation transforms and mild train augmentation."""
    resize_size = int(round(image_size * 1.14))
    if train:
        return v2.Compose(
            [
                v2.Resize(resize_size),
                v2.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                v2.RandomRotation(degrees=10),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.1, contrast=0.1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return v2.Compose(
        [
            v2.Resize(resize_size),
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_datasets(
    data_root: str | Path,
    classes: Sequence[str] = EXPECTED_CLASSES,
    image_size: int = 224,
    smoke_test: bool = False,
    smoke_images_per_class: int = 2,
) -> VisionDatasets:
    """Load train, validation and test splits with validated class mappings."""
    root = Path(data_root)
    expected_mapping = {class_name: index for index, class_name in enumerate(classes)}
    datasets: dict[str, Dataset] = {
        "train": OrderedImageFolder(
            root / "train",
            expected_classes=classes,
            transform=build_transforms(image_size=image_size, train=True),
        ),
        "validation": OrderedImageFolder(
            root / "validation",
            expected_classes=classes,
            transform=build_transforms(image_size=image_size, train=False),
        ),
        "test": OrderedImageFolder(
            root / "test",
            expected_classes=classes,
            transform=build_transforms(image_size=image_size, train=False),
        ),
    }
    for split_name, dataset in datasets.items():
        class_to_idx = _dataset_class_to_idx(dataset)
        if class_to_idx != expected_mapping:
            raise ValueError(
                f"{split_name} class_to_idx mismatch. Expected {expected_mapping}, "
                f"got {class_to_idx}."
            )
    if smoke_test:
        datasets = {
            split_name: subset_per_class(dataset, max_per_class=smoke_images_per_class)
            for split_name, dataset in datasets.items()
        }
    return VisionDatasets(
        train=datasets["train"],
        validation=datasets["validation"],
        test=datasets["test"],
        class_to_idx=expected_mapping,
    )


def create_dataloaders(
    data_root: str | Path,
    classes: Sequence[str] = EXPECTED_CLASSES,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
    smoke_test: bool = False,
) -> VisionDataLoaders:
    """Create dataloaders for all splits using a reproducible train shuffle."""
    datasets = load_datasets(
        data_root=data_root,
        classes=classes,
        image_size=image_size,
        smoke_test=smoke_test,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return VisionDataLoaders(
        train=DataLoader(
            datasets.train,
            shuffle=True,
            generator=generator,
            drop_last=False,
            **common_kwargs,
        ),
        validation=DataLoader(datasets.validation, shuffle=False, **common_kwargs),
        test=DataLoader(datasets.test, shuffle=False, **common_kwargs),
        class_to_idx=datasets.class_to_idx,
        distributions={
            "train": class_distribution(datasets.train, datasets.class_to_idx),
            "validation": class_distribution(datasets.validation, datasets.class_to_idx),
            "test": class_distribution(datasets.test, datasets.class_to_idx),
        },
    )


def class_distribution(dataset: Dataset, class_to_idx: dict[str, int]) -> dict[str, int]:
    """Calculate the number of images per class for ImageFolder or Subset datasets."""
    targets = _dataset_targets(dataset)
    counts = Counter(int(target) for target in targets)
    return {
        class_name: counts.get(index, 0)
        for class_name, index in sorted(class_to_idx.items(), key=lambda item: item[1])
    }


def subset_per_class(dataset: Dataset, max_per_class: int) -> Subset:
    """Return a small deterministic subset with up to N images per class."""
    targets = _dataset_targets(dataset)
    selected: list[int] = []
    counts: Counter[int] = Counter()
    for index, target in enumerate(targets):
        target_index = int(target)
        if counts[target_index] < max_per_class:
            selected.append(index)
            counts[target_index] += 1
    return Subset(dataset, selected)


def _dataset_class_to_idx(dataset: Dataset) -> dict[str, int]:
    if isinstance(dataset, Subset):
        return _dataset_class_to_idx(dataset.dataset)
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if not isinstance(class_to_idx, dict):
        raise TypeError("Dataset does not expose class_to_idx.")
    return {str(class_name): int(index) for class_name, index in class_to_idx.items()}


def _dataset_targets(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        targets = _dataset_targets(dataset.dataset)
        return [int(targets[index]) for index in dataset.indices]
    targets = getattr(dataset, "targets", None)
    if targets is None:
        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise TypeError("Dataset does not expose targets or samples.")
        targets = [sample[1] for sample in samples]
    return [int(target) for target in targets]
