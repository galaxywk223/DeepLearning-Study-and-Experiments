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
        description="Train and evaluate a subword-level GPT language model.",
    )
    parser.add_argument(
        "--experiment-name",
        default=default_config.experiment_name,
        help="Subdirectory name under the output directory.",
    )
    parser.add_argument("--data-dir", default=str(default_config.data_dir))
    parser.add_argument("--output-dir", default=str(default_config.output_dir))
    parser.add_argument(
        "--corpus-filename",
        default=default_config.corpus_filename,
        help="Local text file name stored under data-dir.",
    )
    parser.add_argument(
        "--source-url",
        default=default_config.source_url,
        help="Used only when the local corpus file is missing.",
    )
    parser.add_argument(
        "--tokenizer-filename",
        default=default_config.tokenizer_filename,
        help="Optional tokenizer JSON file name stored under data-dir.",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=default_config.tokenizer_vocab_size,
        help="Target vocabulary size including special tokens.",
    )
    parser.add_argument(
        "--min-pair-frequency",
        type=int,
        default=default_config.min_pair_frequency,
    )
    parser.add_argument("--val-ratio", type=float, default=default_config.val_ratio)
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size)
    parser.add_argument("--epochs", type=int, default=default_config.epochs)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=default_config.steps_per_epoch,
    )
    parser.add_argument("--eval-steps", type=int, default=default_config.eval_steps)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=default_config.grad_accum_steps,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_config.learning_rate,
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=default_config.min_learning_rate,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=default_config.warmup_steps,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=default_config.weight_decay,
    )
    parser.add_argument("--grad-clip", type=float, default=default_config.grad_clip)
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument("--block-size", type=int, default=default_config.block_size)
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=default_config.min_sequence_length,
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=default_config.embedding_dim,
    )
    parser.add_argument("--num-heads", type=int, default=default_config.num_heads)
    parser.add_argument("--num-layers", type=int, default=default_config.num_layers)
    parser.add_argument("--mlp-ratio", type=float, default=default_config.mlp_ratio)
    parser.add_argument("--dropout", type=float, default=default_config.dropout)
    parser.add_argument(
        "--device",
        default=default_config.device,
        help="Use 'auto', 'cpu', or a torch device string such as 'cuda'.",
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=default_config.use_amp,
        help="Enable automatic mixed precision on CUDA.",
    )
    parser.add_argument("--pad-token", default=default_config.pad_token)
    parser.add_argument("--bos-token", default=default_config.bos_token)
    parser.add_argument("--eos-token", default=default_config.eos_token)
    parser.add_argument("--sample-prompt", default=default_config.sample_prompt)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_config.max_new_tokens,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default_config.temperature,
    )
    parser.add_argument("--top-k", type=int, default=default_config.top_k)
    parser.add_argument("--top-p", type=float, default=default_config.top_p)
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the loss curve after saving it.",
    )
    return parser


def config_from_args(variant: str, args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        variant=variant,
        experiment_name=args.experiment_name,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        corpus_filename=args.corpus_filename,
        source_url=args.source_url,
        tokenizer_filename=args.tokenizer_filename,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_steps=args.eval_steps,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        block_size=args.block_size,
        min_sequence_length=args.min_sequence_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        device=args.device,
        use_amp=args.use_amp,
        pad_token=args.pad_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        sample_prompt=args.sample_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        show_plot=args.show_plot,
    )

