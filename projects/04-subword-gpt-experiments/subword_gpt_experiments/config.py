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
    tokenizer_filename: str = ""
    tokenizer_vocab_size: int = 512
    min_pair_frequency: int = 2
    val_ratio: float = 0.1
    batch_size: int = 24
    epochs: int = 8
    steps_per_epoch: int = 180
    eval_steps: int = 40
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    block_size: int = 128
    min_sequence_length: int = 32
    embedding_dim: int = 192
    num_heads: int = 6
    num_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.2
    device: str = "auto"
    use_amp: bool = True
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sample_prompt: str = "ROMEO:\n"
    max_new_tokens: int = 220
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    show_plot: bool = False

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_name

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / self.corpus_filename

    @property
    def resolved_tokenizer_filename(self) -> str:
        if self.tokenizer_filename:
            return self.tokenizer_filename
        stem = Path(self.corpus_filename).stem.replace(" ", "_")
        return f"{stem}-bpe-vocab{self.tokenizer_vocab_size}.json"

    @property
    def tokenizer_path(self) -> Path:
        return self.data_dir / self.resolved_tokenizer_filename

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
            "tokenizer_filename": self.resolved_tokenizer_filename,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "min_pair_frequency": self.min_pair_frequency,
            "val_ratio": self.val_ratio,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "eval_steps": self.eval_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "block_size": self.block_size,
            "min_sequence_length": self.min_sequence_length,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "device": self.device,
            "use_amp": self.use_amp,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "sample_prompt": self.sample_prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
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
            tokenizer_filename=str(payload.get("tokenizer_filename", "")),
            tokenizer_vocab_size=int(payload.get("tokenizer_vocab_size", 512)),
            min_pair_frequency=int(payload.get("min_pair_frequency", 2)),
            val_ratio=float(payload.get("val_ratio", 0.1)),
            batch_size=int(payload.get("batch_size", 24)),
            epochs=int(payload.get("epochs", 8)),
            steps_per_epoch=int(payload.get("steps_per_epoch", 180)),
            eval_steps=int(payload.get("eval_steps", 40)),
            grad_accum_steps=int(payload.get("grad_accum_steps", 1)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            min_learning_rate=float(payload.get("min_learning_rate", 3e-5)),
            warmup_steps=int(payload.get("warmup_steps", 100)),
            weight_decay=float(payload.get("weight_decay", 0.1)),
            grad_clip=float(payload.get("grad_clip", 1.0)),
            seed=int(payload.get("seed", 42)),
            block_size=int(payload.get("block_size", 128)),
            min_sequence_length=int(payload.get("min_sequence_length", 32)),
            embedding_dim=int(payload.get("embedding_dim", 192)),
            num_heads=int(payload.get("num_heads", 6)),
            num_layers=int(payload.get("num_layers", 6)),
            mlp_ratio=float(payload.get("mlp_ratio", 4.0)),
            dropout=float(payload.get("dropout", 0.2)),
            device=str(payload.get("device", "auto")),
            use_amp=bool(payload.get("use_amp", True)),
            pad_token=str(payload.get("pad_token", "<pad>")),
            bos_token=str(payload.get("bos_token", "<bos>")),
            eos_token=str(payload.get("eos_token", "<eos>")),
            sample_prompt=str(payload.get("sample_prompt", "ROMEO:\n")),
            max_new_tokens=int(payload.get("max_new_tokens", 220)),
            temperature=float(payload.get("temperature", 0.8)),
            top_k=int(payload.get("top_k", 40)),
            top_p=float(payload.get("top_p", 0.95)),
            show_plot=bool(payload.get("show_plot", False)),
        )


def create_default_config(variant: str) -> ExperimentConfig:
    normalized = variant.lower()
    if normalized != "gpt":
        raise ValueError(f"Unsupported variant: {variant}")

    return ExperimentConfig(
        variant="gpt",
        experiment_name="tinyshakespeare-subword-gpt",
        tokenizer_vocab_size=512,
        min_pair_frequency=2,
        batch_size=24,
        epochs=8,
        steps_per_epoch=180,
        eval_steps=40,
        grad_accum_steps=1,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.1,
        grad_clip=1.0,
        block_size=128,
        min_sequence_length=32,
        embedding_dim=192,
        num_heads=6,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.2,
        use_amp=True,
        top_k=40,
        top_p=0.95,
    )

