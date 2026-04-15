from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_TEMPLATE_GROUP, TEMPLATE_GROUPS
from .utils import read_json, write_text


def run_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    markdown = summarize_run_dirs([Path(run_dir) for run_dir in args.run_dir])
    if args.output:
        write_text(Path(args.output), markdown)
        print(f"Wrote summary: {args.output}")
        return
    print(markdown)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize multiple Notes Assistant template-group runs.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Experiment output directory. Repeat this flag for multiple runs.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional markdown output path.",
    )
    return parser


def summarize_run_dirs(run_dirs: list[Path]) -> str:
    rows: list[dict[str, object]] = []
    template_breakdowns: dict[str, dict[str, dict[str, object]]] = {}
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        train_metrics_path = run_dir / "metrics.json"
        eval_metrics_path = run_dir / "evaluation" / "metrics.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        if not train_metrics_path.exists():
            raise FileNotFoundError(f"Missing training metrics: {train_metrics_path}")
        if not eval_metrics_path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics: {eval_metrics_path}")

        config = read_json(config_path)
        train_metrics = read_json(train_metrics_path)
        eval_metrics = read_json(eval_metrics_path)
        experiment_name = str(config["experiment_name"])
        template_group = str(config.get("template_group", DEFAULT_TEMPLATE_GROUP))
        selected_template_ids = config.get("selected_template_ids")
        if not selected_template_ids:
            selected_template_ids = TEMPLATE_GROUPS.get(template_group, ())
        rows.append(
            {
                "experiment_name": experiment_name,
                "template_group": template_group,
                "selected_template_ids": ", ".join(
                    str(item) for item in selected_template_ids
                ),
                "train_count": train_metrics["split_counts"]["train"],
                "val_count": train_metrics["split_counts"]["val"],
                "test_count": train_metrics["split_counts"]["test"],
                "base_avg_char_f1": eval_metrics["base_avg_char_f1"],
                "finetuned_avg_char_f1": eval_metrics["finetuned_avg_char_f1"],
                "tuned_better_rate": eval_metrics["tuned_better_rate"],
            }
        )
        template_breakdowns[experiment_name] = dict(
            sorted(eval_metrics.get("by_template_id", {}).items())
        )

    rows.sort(
        key=lambda item: (
            str(item["template_group"]),
            str(item["experiment_name"]),
        )
    )
    return render_markdown(rows, template_breakdowns)


def render_markdown(
    rows: list[dict[str, object]],
    template_breakdowns: dict[str, dict[str, dict[str, object]]],
) -> str:
    lines = [
        "# Notes Assistant Template Group Ablation",
        "",
        "## Overall Comparison",
        "",
        "| Experiment | Template Group | Selected Templates | Train / Val / Test | Base F1 | Tuned F1 | Tuned Better Rate |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['experiment_name']} | "
            f"{row['template_group']} | "
            f"{row['selected_template_ids']} | "
            f"{row['train_count']} / {row['val_count']} / {row['test_count']} | "
            f"{float(row['base_avg_char_f1']):.4f} | "
            f"{float(row['finetuned_avg_char_f1']):.4f} | "
            f"{float(row['tuned_better_rate']):.2%} |"
        )

    lines.extend(["", "## Breakdown by Template", ""])
    for row in rows:
        experiment_name = str(row["experiment_name"])
        lines.extend(
            [
                f"### {experiment_name}",
                "",
                "| Template | Sample Count | Base F1 | Tuned F1 | Tuned Better Rate |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for template_id, metrics in template_breakdowns[experiment_name].items():
            lines.append(
                "| "
                f"{template_id} | "
                f"{int(metrics['sample_count'])} | "
                f"{float(metrics['base_avg_char_f1']):.4f} | "
                f"{float(metrics['finetuned_avg_char_f1']):.4f} | "
                f"{float(metrics['tuned_better_rate']):.2%} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"
