from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .utils import read_json


def run_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config_from_args(args)
    launch_demo(
        config,
        run_dir=Path(args.run_dir),
        adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
        host=args.host,
        port=args.port,
        share=args.share,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a local Gradio demo for the notes assistant.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Experiment output directory containing config.json and adapter/.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="",
        help="Override the adapter directory if needed.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser


def load_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    payload = read_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected config payload in {config_path}")
    return ExperimentConfig.from_dict(payload)


def launch_demo(
    config: ExperimentConfig,
    *,
    run_dir: Path,
    adapter_dir: Path | None,
    host: str,
    port: int,
    share: bool,
) -> None:
    import gradio as gr

    from .data import compose_user_message, load_sft_splits
    from .inference import generate_answer, load_model_for_inference, load_tokenizer

    resolved_adapter_dir = adapter_dir or (run_dir / "adapter")
    tokenizer = load_tokenizer(str(resolved_adapter_dir))
    model = load_model_for_inference(
        model_id=config.model_id,
        adapter_dir=resolved_adapter_dir,
        quantization_mode=config.quantization_mode,
        device_map=config.device_map,
    )
    data_bundle = load_sft_splits(config, tokenizer, overwrite_dataset=False)
    example_questions = [
        compose_user_message(record)
        for record in data_bundle.test_records[:5]
    ]

    def respond(message: str, history) -> str:
        return generate_answer(
            model,
            tokenizer,
            system_prompt=config.system_prompt,
            user_message=message,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )

    interface = gr.ChatInterface(
        fn=respond,
        title="DeepLearning Notes Assistant",
        description=(
            "基于当前仓库笔记构建的 LoRA 助教。"
            "它只应该回答现有章节覆盖的内容。"
        ),
        examples=example_questions,
    )
    interface.launch(server_name=host, server_port=port, share=share)
