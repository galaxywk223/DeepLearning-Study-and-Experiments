from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, create_default_config


def run_cli(variant: str) -> None:
    parser = build_parser(variant)
    args = parser.parse_args()
    config = config_from_args(variant, args)

    from .runner import run_experiment

    run_experiment(config)


def build_parser(variant: str) -> argparse.ArgumentParser:
    default_config = create_default_config(variant)
    parser = argparse.ArgumentParser(
        description=f"Train and evaluate the CIFAR-10 {variant} experiment.",
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
        "--label-smoothing",
        type=float,
        default=default_config.label_smoothing,
    )
    parser.add_argument("--dropout", type=float, default=default_config.dropout)
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=default_config.use_amp,
        help="Enable automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--scheduler",
        default=default_config.scheduler,
        help="Supported: none, cosine, multistep.",
    )
    parser.add_argument(
        "--optimizer-name",
        default=default_config.optimizer_name,
        help="Supported: adam, adamw, sgd.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=default_config.momentum,
    )
    parser.add_argument(
        "--random-crop-padding",
        type=int,
        default=default_config.random_crop_padding,
    )
    parser.add_argument(
        "--random-horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=default_config.random_horizontal_flip,
    )
    parser.add_argument(
        "--color-jitter-strength",
        type=float,
        default=default_config.color_jitter_strength,
    )
    parser.add_argument(
        "--random-erasing-prob",
        type=float,
        default=default_config.random_erasing_prob,
    )
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


def config_from_args(variant: str, args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        variant=variant,
        experiment_name=args.experiment_name,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        use_amp=args.use_amp,
        scheduler=args.scheduler,
        optimizer_name=args.optimizer_name,
        momentum=args.momentum,
        random_crop_padding=args.random_crop_padding,
        random_horizontal_flip=args.random_horizontal_flip,
        color_jitter_strength=args.color_jitter_strength,
        random_erasing_prob=args.random_erasing_prob,
        device=args.device,
        preview_count=args.preview_count,
        show_plot=args.show_plot,
    )
