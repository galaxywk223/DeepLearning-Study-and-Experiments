from __future__ import annotations

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .config import ExperimentConfig

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def build_mnist_dataloaders(
    config: ExperimentConfig,
    *,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    train_transform = build_train_transform(config.rotation_degrees)
    test_transform = build_eval_transform()

    train_set = datasets.MNIST(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.MNIST(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def build_train_transform(rotation_degrees: float) -> transforms.Compose:
    steps: list[object] = []
    if rotation_degrees > 0:
        steps.append(transforms.RandomRotation(rotation_degrees))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
