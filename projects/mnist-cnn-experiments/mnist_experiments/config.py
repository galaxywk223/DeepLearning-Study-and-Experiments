from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExperimentConfig:
    model_name: str
    experiment_name: str
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 42
    rotation_degrees: float = 0.0
    hidden_dim: int = 128
    dropout: float = 0.5
    device: str = "auto"
    preview_count: int = 5
    show_plot: bool = False

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
            "model_name": self.model_name,
            "experiment_name": self.experiment_name,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "rotation_degrees": self.rotation_degrees,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "device": self.device,
            "preview_count": self.preview_count,
            "show_plot": self.show_plot,
        }


def create_default_config(model_name: str) -> ExperimentConfig:
    normalized = model_name.lower()
    if normalized == "mlp":
        return ExperimentConfig(
            model_name="mlp",
            experiment_name="mlp-baseline",
            epochs=10,
            learning_rate=1e-2,
            hidden_dim=128,
        )
    if normalized == "cnn":
        return ExperimentConfig(
            model_name="cnn",
            experiment_name="cnn-improved",
            epochs=15,
            learning_rate=1e-3,
            weight_decay=1e-4,
            rotation_degrees=5.0,
            hidden_dim=512,
            dropout=0.5,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
