from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ExperimentConfig


def run_generation_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    temperatures = [float(value) for value in args.temperatures]
    generate_temperature_sweep(
        run_dir=Path(args.run_dir),
        prompt=args.prompt,
        temperatures=temperatures,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
        top_k=args.top_k,
        top_p=args.top_p,
        output_path=Path(args.output_path) if args.output_path else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained subword GPT model at multiple temperatures.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed experiment directory containing config.json and best_model.pt.",
    )
    parser.add_argument(
        "--prompt",
        default="ROMEO:\n",
        help="Prompt used for all generated samples.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        default=["0.6", "0.8", "1.0"],
        help="One or more sampling temperatures.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of generations per temperature.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=320,
        help="Number of new tokens to sample per generation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', or a torch device string such as 'cuda'.",
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional output text file. Defaults to <run-dir>/temperature_sweep.txt.",
    )
    return parser


def generate_temperature_sweep(
    *,
    run_dir: Path,
    prompt: str,
    temperatures: list[float],
    num_samples: int,
    max_new_tokens: int,
    device_name: str,
    top_k: int,
    top_p: float,
    output_path: Path | None,
) -> str:
    from .data import prepare_dataset
    from .models import build_model
    from .utils import resolve_device, write_text

    config = load_config_from_run(run_dir)
    dataset = prepare_dataset(config)
    tokenizer = dataset.tokenizer
    device = resolve_device(device_name)

    model = build_model(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        config=config,
    ).to(device)
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    prompt_tokens = tokenizer.encode(prompt or "\n", add_bos=True)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    sections = [
        f"Run: {run_dir}",
        f"Prompt:\n{prompt}",
        f"Device: {device.type}",
        f"Top-k: {top_k}",
        f"Top-p: {top_p}",
        "",
    ]

    for temperature in temperatures:
        sections.append(f"Temperature: {temperature}")
        sections.append("-" * 40)
        for sample_index in range(1, num_samples + 1):
            with torch.no_grad():
                generated_tokens = model.generate(
                    prompt_tensor.clone(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    allowed_token_ids=dataset.observed_token_ids,
                )
            generated_text = tokenizer.decode(generated_tokens[0].cpu().tolist())
            sections.append(f"[Sample {sample_index}]")
            sections.append(generated_text)
            sections.append("")
        sections.append("")

    report = "\n".join(sections).rstrip() + "\n"
    destination = output_path or (run_dir / "temperature_sweep.txt")
    write_text(destination, report)
    print(f"Saved temperature sweep to {destination}")
    return report


def load_config_from_run(run_dir: Path) -> ExperimentConfig:
    config_path = run_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig.from_dict(payload)
