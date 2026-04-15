from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SYSTEM_PROMPT = (
    "你是一名深度学习学习助教，只回答当前课程笔记覆盖的内容；"
    "不确定时明确说不知道，不编造未覆盖细节。"
)
DEFAULT_TEMPLATE_IDS = (
    "explain_core",
    "key_points",
    "study_focus",
    "chapter_position",
    "experiment_bridge",
)
TEMPLATE_GROUPS = {
    "full": DEFAULT_TEMPLATE_IDS,
    "content": (
        "explain_core",
        "key_points",
        "study_focus",
    ),
    "structure": (
        "explain_core",
        "chapter_position",
        "experiment_bridge",
    ),
    "core_only": ("explain_core",),
}
DEFAULT_TEMPLATE_GROUP = "full"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str = "notes-assistant-qwen25-0p5b"
    notes_dir: Path = REPO_ROOT / "notes"
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    dataset_filename: str = "notes-assistant-qa.jsonl"
    dataset_summary_filename: str = "notes-assistant-dataset-summary.json"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    template_group: str = DEFAULT_TEMPLATE_GROUP
    max_seq_length: int = 512
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    epochs: float = 3.0
    max_train_samples: int = 0
    max_eval_samples: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_TARGET_MODULES
    )
    seed: int = 42
    device_map: str = "auto"
    quantization_mode: str = "4bit"
    logging_steps: int = 5
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset_filename

    @property
    def dataset_summary_path(self) -> Path:
        return self.data_dir / self.dataset_summary_filename

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_name

    @property
    def adapter_dir(self) -> Path:
        return self.run_dir / "adapter"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def loss_curve_path(self) -> Path:
        return self.run_dir / "loss_curve.png"

    @property
    def sample_path(self) -> Path:
        return self.run_dir / "samples.md"

    @property
    def comparison_path(self) -> Path:
        return self.run_dir / "comparison_samples.md"

    @property
    def evaluation_dir(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def selected_template_ids(self) -> tuple[str, ...]:
        try:
            return TEMPLATE_GROUPS[self.template_group]
        except KeyError as exc:
            valid = ", ".join(sorted(TEMPLATE_GROUPS))
            raise ValueError(
                f"Unknown template group '{self.template_group}'. Expected one of: {valid}"
            ) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "notes_dir": str(self.notes_dir),
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "dataset_filename": self.dataset_filename,
            "dataset_summary_filename": self.dataset_summary_filename,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
            "template_group": self.template_group,
            "selected_template_ids": list(self.selected_template_ids),
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "epochs": self.epochs,
            "max_train_samples": self.max_train_samples,
            "max_eval_samples": self.max_eval_samples,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "seed": self.seed,
            "device_map": self.device_map,
            "quantization_mode": self.quantization_mode,
            "logging_steps": self.logging_steps,
            "save_total_limit": self.save_total_limit,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_name=str(
                payload.get("experiment_name", "notes-assistant-qwen25-0p5b")
            ),
            notes_dir=Path(str(payload.get("notes_dir", REPO_ROOT / "notes"))),
            data_dir=Path(str(payload.get("data_dir", PROJECT_ROOT / "data"))),
            output_dir=Path(str(payload.get("output_dir", PROJECT_ROOT / "outputs"))),
            dataset_filename=str(
                payload.get("dataset_filename", "notes-assistant-qa.jsonl")
            ),
            dataset_summary_filename=str(
                payload.get(
                    "dataset_summary_filename",
                    "notes-assistant-dataset-summary.json",
                )
            ),
            model_id=str(payload.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")),
            system_prompt=str(payload.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
            template_group=str(
                payload.get("template_group", DEFAULT_TEMPLATE_GROUP)
            ),
            max_seq_length=int(payload.get("max_seq_length", 512)),
            per_device_train_batch_size=int(
                payload.get("per_device_train_batch_size", 1)
            ),
            per_device_eval_batch_size=int(
                payload.get("per_device_eval_batch_size", 1)
            ),
            grad_accum_steps=int(payload.get("grad_accum_steps", 16)),
            learning_rate=float(payload.get("learning_rate", 2e-4)),
            weight_decay=float(payload.get("weight_decay", 0.01)),
            warmup_ratio=float(payload.get("warmup_ratio", 0.03)),
            epochs=float(payload.get("epochs", 3.0)),
            max_train_samples=int(payload.get("max_train_samples", 0)),
            max_eval_samples=int(payload.get("max_eval_samples", 0)),
            lora_r=int(payload.get("lora_r", 16)),
            lora_alpha=int(payload.get("lora_alpha", 32)),
            lora_dropout=float(payload.get("lora_dropout", 0.05)),
            target_modules=tuple(
                str(item)
                for item in payload.get("target_modules", DEFAULT_TARGET_MODULES)
            ),
            seed=int(payload.get("seed", 42)),
            device_map=str(payload.get("device_map", "auto")),
            quantization_mode=str(payload.get("quantization_mode", "4bit")),
            logging_steps=int(payload.get("logging_steps", 5)),
            save_total_limit=int(payload.get("save_total_limit", 2)),
            gradient_checkpointing=bool(
                payload.get("gradient_checkpointing", True)
            ),
            max_new_tokens=int(payload.get("max_new_tokens", 256)),
            temperature=float(payload.get("temperature", 0.2)),
            top_p=float(payload.get("top_p", 0.9)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.05)),
        )


def create_default_config() -> ExperimentConfig:
    return ExperimentConfig()
