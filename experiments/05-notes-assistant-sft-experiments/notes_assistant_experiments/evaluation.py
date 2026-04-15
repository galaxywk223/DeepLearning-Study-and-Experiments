from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .utils import char_level_f1, exact_match, read_json, write_csv, write_json, write_jsonl, write_text


def run_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config_from_args(args)
    run_evaluation(
        config,
        run_dir=Path(args.run_dir),
        adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
        max_test_samples=args.max_test_samples,
        overwrite_dataset=args.rebuild_dataset,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the base model and the fine-tuned adapter side by side.",
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
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
    )
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


def run_evaluation(
    config: ExperimentConfig,
    *,
    run_dir: Path,
    adapter_dir: Path | None,
    max_test_samples: int,
    overwrite_dataset: bool,
) -> dict[str, object]:
    from .data import compose_user_message, load_sft_splits
    from .inference import (
        generate_answer,
        load_model_for_inference,
        load_tokenizer,
        unload_model,
    )

    resolved_adapter_dir = adapter_dir or (run_dir / "adapter")
    if not resolved_adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter directory: {resolved_adapter_dir}")

    tokenizer = load_tokenizer(str(resolved_adapter_dir))
    data_bundle = load_sft_splits(
        config,
        tokenizer,
        overwrite_dataset=overwrite_dataset,
    )
    test_records = data_bundle.test_records
    if max_test_samples > 0:
        test_records = test_records[:max_test_samples]

    base_model = load_model_for_inference(
        model_id=config.model_id,
        adapter_dir=None,
        quantization_mode=config.quantization_mode,
        device_map=config.device_map,
    )
    base_predictions = [
        generate_answer(
            base_model,
            tokenizer,
            system_prompt=config.system_prompt,
            user_message=compose_user_message(record),
            max_new_tokens=config.max_new_tokens,
            temperature=0.0,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )
        for record in test_records
    ]
    unload_model(base_model)

    tuned_model = load_model_for_inference(
        model_id=config.model_id,
        adapter_dir=resolved_adapter_dir,
        quantization_mode=config.quantization_mode,
        device_map=config.device_map,
    )
    tuned_predictions = [
        generate_answer(
            tuned_model,
            tokenizer,
            system_prompt=config.system_prompt,
            user_message=compose_user_message(record),
            max_new_tokens=config.max_new_tokens,
            temperature=0.0,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )
        for record in test_records
    ]
    unload_model(tuned_model)

    detailed_rows: list[dict[str, object]] = []
    for record, base_answer, tuned_answer in zip(
        test_records,
        base_predictions,
        tuned_predictions,
        strict=True,
    ):
        reference = str(record["output"])
        detailed_rows.append(
            {
                "id": record["id"],
                "source_chapter": record["source_chapter"],
                "source_section": record["source_section"],
                "instruction": record["instruction"],
                "input": record["input"],
                "reference": reference,
                "base_answer": base_answer,
                "finetuned_answer": tuned_answer,
                "base_char_f1": char_level_f1(base_answer, reference),
                "finetuned_char_f1": char_level_f1(tuned_answer, reference),
                "base_exact_match": exact_match(base_answer, reference),
                "finetuned_exact_match": exact_match(tuned_answer, reference),
            }
        )

    metrics = summarize_metrics(detailed_rows)
    evaluation_dir = config.evaluation_dir
    predictions_path = evaluation_dir / "predictions.jsonl"
    report_path = evaluation_dir / "report.md"
    review_path = evaluation_dir / "manual_review_template.csv"
    metrics_path = evaluation_dir / "metrics.json"

    write_jsonl(predictions_path, detailed_rows)
    write_json(metrics_path, metrics)
    write_text(report_path, render_report(detailed_rows, metrics))
    write_csv(
        review_path,
        build_manual_review_rows(detailed_rows),
        fieldnames=manual_review_fields(),
    )
    return metrics


def summarize_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"sample_count": 0}

    base_f1 = sum(float(row["base_char_f1"]) for row in rows) / len(rows)
    tuned_f1 = sum(float(row["finetuned_char_f1"]) for row in rows) / len(rows)
    base_em = sum(float(row["base_exact_match"]) for row in rows) / len(rows)
    tuned_em = sum(float(row["finetuned_exact_match"]) for row in rows) / len(rows)
    tuned_better = sum(
        float(row["finetuned_char_f1"]) > float(row["base_char_f1"]) for row in rows
    )
    return {
        "sample_count": len(rows),
        "base_avg_char_f1": base_f1,
        "finetuned_avg_char_f1": tuned_f1,
        "base_avg_exact_match": base_em,
        "finetuned_avg_exact_match": tuned_em,
        "tuned_better_count": tuned_better,
        "tuned_better_rate": tuned_better / len(rows),
    }


def render_report(
    rows: list[dict[str, object]],
    metrics: dict[str, object],
) -> str:
    lines = [
        "# Notes Assistant Evaluation",
        "",
        f"- Sample count: {metrics['sample_count']}",
        f"- Base avg char F1: {metrics['base_avg_char_f1']:.4f}",
        f"- Finetuned avg char F1: {metrics['finetuned_avg_char_f1']:.4f}",
        f"- Base exact match: {metrics['base_avg_exact_match']:.4f}",
        f"- Finetuned exact match: {metrics['finetuned_avg_exact_match']:.4f}",
        f"- Finetuned better rate: {metrics['tuned_better_rate']:.2%}",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Example {index}",
                "",
                f"**Source**: {row['source_chapter']} / {row['source_section']}",
                "",
                "**Question**",
                "",
                str(row["instruction"]),
                "",
                "**Input**",
                "",
                str(row["input"]),
                "",
                "**Reference**",
                "",
                str(row["reference"]),
                "",
                f"**Base Answer** (`char_f1={row['base_char_f1']:.4f}`)",
                "",
                str(row["base_answer"]),
                "",
                f"**Finetuned Answer** (`char_f1={row['finetuned_char_f1']:.4f}`)",
                "",
                str(row["finetuned_answer"]),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def manual_review_fields() -> list[str]:
    return [
        "id",
        "source_chapter",
        "source_section",
        "instruction",
        "reference",
        "base_answer",
        "finetuned_answer",
        "base_relevance",
        "base_correctness",
        "base_terminology",
        "finetuned_relevance",
        "finetuned_correctness",
        "finetuned_terminology",
        "winner",
        "notes",
    ]


def build_manual_review_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    review_rows: list[dict[str, object]] = []
    for row in rows:
        review_rows.append(
            {
                "id": row["id"],
                "source_chapter": row["source_chapter"],
                "source_section": row["source_section"],
                "instruction": row["instruction"],
                "reference": row["reference"],
                "base_answer": row["base_answer"],
                "finetuned_answer": row["finetuned_answer"],
                "base_relevance": "",
                "base_correctness": "",
                "base_terminology": "",
                "finetuned_relevance": "",
                "finetuned_correctness": "",
                "finetuned_terminology": "",
                "winner": "",
                "notes": "",
            }
        )
    return review_rows

