from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ExperimentConfig:
    variant: str
    experiment_name: str
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    corpus_filename: str = "tiny_shakespeare.txt"
    source_url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt"
    )
    batch_size: int = 64
    epochs: int = 8
    steps_per_epoch: int = 200
    eval_steps: int = 40
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    block_size: int = 128
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.2
    device: str = "auto"
    use_amp: bool = True
    sample_prompt: str = "ROMEO:\n"
    max_new_tokens: int = 300
    temperature: float = 0.8
    show_plot: bool = False

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_name

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / self.corpus_filename

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
    def sample_path(self) -> Path:
        return self.run_dir / "samples.txt"

    @property
    def loss_curve_path(self) -> Path:
        return self.run_dir / "loss_curve.png"

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "experiment_name": self.experiment_name,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "corpus_filename": self.corpus_filename,
            "source_url": self.source_url,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "eval_steps": self.eval_steps,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "block_size": self.block_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "device": self.device,
            "use_amp": self.use_amp,
            "sample_prompt": self.sample_prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "show_plot": self.show_plot,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            variant=str(payload["variant"]),
            experiment_name=str(payload["experiment_name"]),
            data_dir=Path(str(payload.get("data_dir", PROJECT_ROOT / "data"))),
            output_dir=Path(str(payload.get("output_dir", PROJECT_ROOT / "outputs"))),
            corpus_filename=str(payload.get("corpus_filename", "tiny_shakespeare.txt")),
            source_url=str(payload.get("source_url", "")),
            batch_size=int(payload.get("batch_size", 64)),
            epochs=int(payload.get("epochs", 8)),
            steps_per_epoch=int(payload.get("steps_per_epoch", 200)),
            eval_steps=int(payload.get("eval_steps", 40)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            min_learning_rate=float(payload.get("min_learning_rate", 3e-5)),
            warmup_steps=int(payload.get("warmup_steps", 100)),
            weight_decay=float(payload.get("weight_decay", 0.1)),
            grad_clip=float(payload.get("grad_clip", 1.0)),
            seed=int(payload.get("seed", 42)),
            block_size=int(payload.get("block_size", 128)),
            embedding_dim=int(payload.get("embedding_dim", 128)),
            num_heads=int(payload.get("num_heads", 4)),
            num_layers=int(payload.get("num_layers", 4)),
            mlp_ratio=float(payload.get("mlp_ratio", 4.0)),
            dropout=float(payload.get("dropout", 0.2)),
            device=str(payload.get("device", "auto")),
            use_amp=bool(payload.get("use_amp", True)),
            sample_prompt=str(payload.get("sample_prompt", "ROMEO:\n")),
            max_new_tokens=int(payload.get("max_new_tokens", 300)),
            temperature=float(payload.get("temperature", 0.8)),
            show_plot=bool(payload.get("show_plot", False)),
        )


def create_default_config(variant: str) -> ExperimentConfig:
    normalized = variant.lower()
    if normalized == "bigram":
        return ExperimentConfig(
            variant="bigram",
            experiment_name="tinyshakespeare-bigram",
            epochs=4,
            steps_per_epoch=120,
            eval_steps=20,
            learning_rate=1e-2,
            min_learning_rate=1e-2,
            warmup_steps=0,
            weight_decay=0.0,
            grad_clip=0.0,
            block_size=64,
            embedding_dim=64,
            num_heads=1,
            num_layers=1,
            mlp_ratio=1.0,
            dropout=0.0,
            use_amp=False,
            temperature=1.0,
        )
    if normalized == "transformer":
        return ExperimentConfig(
            variant="transformer",
            experiment_name="tinyshakespeare-transformer",
            epochs=8,
            steps_per_epoch=200,
            eval_steps=40,
            learning_rate=3e-4,
            min_learning_rate=3e-5,
            warmup_steps=100,
            weight_decay=0.1,
            grad_clip=1.0,
            block_size=128,
            embedding_dim=128,
            num_heads=4,
            num_layers=4,
            mlp_ratio=4.0,
            dropout=0.2,
            use_amp=True,
            temperature=0.8,
        )
    raise ValueError(f"Unsupported variant: {variant}")
