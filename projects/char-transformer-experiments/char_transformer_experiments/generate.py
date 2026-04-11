from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig


def run_generation_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    temperatures = [float(value) for value in args.temperatures]
    generate_temperature_sweep(
        run_dir=run_dir,
        prompt=args.prompt,
        temperatures=temperatures,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
        output_path=Path(args.output_path) if args.output_path else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained char-level model at multiple temperatures.",
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
        default=400,
        help="Number of new tokens to sample per generation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', or a torch device string such as 'cuda'.",
    )
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
    output_path: Path | None,
) -> str:
    import torch

    from .data import decode_tokens, encode_text, prepare_dataset
    from .models import build_model
    from .runner import sanitize_prompt
    from .utils import resolve_device, write_text

    config = load_config_from_run(run_dir)
    dataset = prepare_dataset(config)
    device = resolve_device(device_name)

    model = build_model(config.variant, vocab_size=dataset.vocab_size, config=config).to(device)
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    cleaned_prompt = sanitize_prompt(prompt, dataset.stoi)
    prompt_tokens = encode_text(cleaned_prompt, dataset.stoi).unsqueeze(0).to(device)

    sections = [
        f"Run: {run_dir}",
        f"Prompt:\n{cleaned_prompt}",
        f"Device: {device.type}",
        "",
    ]

    for temperature in temperatures:
        sections.append(f"Temperature: {temperature}")
        sections.append("-" * 40)
        for sample_index in range(1, num_samples + 1):
            with torch.no_grad():
                generated_tokens = model.generate(
                    prompt_tokens.clone(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            generated_text = decode_tokens(generated_tokens[0].cpu(), dataset.itos)
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
