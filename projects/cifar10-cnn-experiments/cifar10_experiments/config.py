from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ExperimentConfig:
    variant: str
    experiment_name: str
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"
    preview_count: int = 8
    show_plot: bool = False
    label_smoothing: float = 0.0
    dropout: float = 0.0
    use_amp: bool = False
    scheduler: str = "none"
    optimizer_name: str = "adam"
    momentum: float = 0.0
    random_crop_padding: int = 0
    random_horizontal_flip: bool = False
    color_jitter_strength: float = 0.0
    random_erasing_prob: float = 0.0

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_name

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "best_model.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def preview_path(self) -> Path:
        return self.run_dir / "predictions.png"

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "experiment_name": self.experiment_name,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "device": self.device,
            "preview_count": self.preview_count,
            "show_plot": self.show_plot,
            "label_smoothing": self.label_smoothing,
            "dropout": self.dropout,
            "use_amp": self.use_amp,
            "scheduler": self.scheduler,
            "optimizer_name": self.optimizer_name,
            "momentum": self.momentum,
            "random_crop_padding": self.random_crop_padding,
            "random_horizontal_flip": self.random_horizontal_flip,
            "color_jitter_strength": self.color_jitter_strength,
            "random_erasing_prob": self.random_erasing_prob,
        }


def create_default_config(variant: str) -> ExperimentConfig:
    normalized = variant.lower()
    if normalized == "baseline":
        return ExperimentConfig(
            variant="baseline",
            experiment_name="cifar10-cnn-baseline",
            epochs=20,
            batch_size=128,
            learning_rate=1e-3,
            num_workers=2,
            optimizer_name="adam",
        )
    if normalized == "improved":
        return ExperimentConfig(
            variant="improved",
            experiment_name="cifar10-cnn-improved",
            epochs=30,
            batch_size=128,
            learning_rate=3e-4,
            weight_decay=5e-4,
            num_workers=4,
            label_smoothing=0.1,
            dropout=0.4,
            use_amp=True,
            scheduler="cosine",
            optimizer_name="adamw",
            random_crop_padding=4,
            random_horizontal_flip=True,
            color_jitter_strength=0.1,
        )
    if normalized == "resnet":
        return ExperimentConfig(
            variant="resnet",
            experiment_name="cifar10-cnn-resnet",
            epochs=100,
            batch_size=128,
            learning_rate=0.1,
            weight_decay=5e-4,
            num_workers=4,
            label_smoothing=0.1,
            dropout=0.0,
            use_amp=True,
            scheduler="multistep",
            optimizer_name="sgd",
            momentum=0.9,
            random_crop_padding=4,
            random_horizontal_flip=True,
            color_jitter_strength=0.1,
            random_erasing_prob=0.25,
        )
    raise ValueError(f"Unsupported variant: {variant}")
