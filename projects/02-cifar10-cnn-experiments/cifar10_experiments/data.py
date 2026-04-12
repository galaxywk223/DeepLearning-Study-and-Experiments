from __future__ import annotations

import platform

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .config import ExperimentConfig

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def build_cifar10_dataloaders(
    config: ExperimentConfig,
    *,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, dict[str, int | bool | str]]:
    train_transform = build_train_transform(config)
    test_transform = build_eval_transform()

    train_set = datasets.CIFAR10(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    effective_num_workers = resolve_num_workers(config.num_workers)
    loader_kwargs: dict[str, int | bool] = {
        "batch_size": config.batch_size,
        "num_workers": effective_num_workers,
        "pin_memory": pin_memory,
    }
    if effective_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    loader_info: dict[str, int | bool | str] = {
        "platform": platform.system(),
        "requested_num_workers": config.num_workers,
        "effective_num_workers": effective_num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": effective_num_workers > 0,
        "prefetch_factor": 2 if effective_num_workers > 0 else 0,
    }
    return train_loader, test_loader, loader_info


def build_train_transform(config: ExperimentConfig) -> transforms.Compose:
    steps: list[object] = []
    if config.random_crop_padding > 0:
        steps.append(transforms.RandomCrop(32, padding=config.random_crop_padding))
    if config.random_horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    if config.color_jitter_strength > 0:
        strength = config.color_jitter_strength
        steps.append(
            transforms.ColorJitter(
                brightness=strength,
                contrast=strength,
                saturation=strength,
                hue=min(0.5 * strength, 0.05),
            )
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    if config.random_erasing_prob > 0:
        steps.append(
            transforms.RandomErasing(
                p=config.random_erasing_prob,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    return transforms.Compose(steps)


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def resolve_num_workers(requested_num_workers: int) -> int:
    requested = max(0, requested_num_workers)
    if platform.system() == "Windows":
        return min(requested, 4)
    return requested
