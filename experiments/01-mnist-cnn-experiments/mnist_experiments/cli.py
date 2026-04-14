from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, create_default_config


def run_cli(model_name: str) -> None:
    parser = build_parser(model_name)
    args = parser.parse_args()
    config = config_from_args(model_name, args)
    from .runner import run_experiment

    run_experiment(config)


def build_parser(model_name: str) -> argparse.ArgumentParser:
    default_config = create_default_config(model_name)
    parser = argparse.ArgumentParser(
        description=f"Train and evaluate the {model_name.upper()} MNIST experiment.",
    )
    parser.add_argument(
        "--experiment-name",
        default=default_config.experiment_name,
        help="Subdirectory name under the output directory.",
    )
    parser.add_argument("--data-dir", default=str(default_config.data_dir))
    parser.add_argument("--output-dir", default=str(default_config.output_dir))
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size)
    parser.add_argument("--epochs", type=int, default=default_config.epochs)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_config.learning_rate,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=default_config.weight_decay,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_config.num_workers,
    )
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=default_config.rotation_degrees,
        help="Applied only to the training split.",
    )
    parser.add_argument("--hidden-dim", type=int, default=default_config.hidden_dim)
    parser.add_argument("--dropout", type=float, default=default_config.dropout)
    parser.add_argument(
        "--device",
        default=default_config.device,
        help="Use 'auto', 'cpu', or a torch device string such as 'cuda'.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=default_config.preview_count,
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the preview figure after saving it.",
    )
    return parser


def config_from_args(model_name: str, args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        model_name=model_name,
        experiment_name=args.experiment_name,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        rotation_degrees=args.rotation_degrees,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=args.device,
        preview_count=args.preview_count,
        show_plot=args.show_plot,
    )
