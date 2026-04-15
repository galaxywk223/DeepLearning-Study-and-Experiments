from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ExperimentConfig, create_default_config
from .utils import normalize_text, read_json, slugify, stable_hash, write_json, write_jsonl

SECTIONS_PER_CHAPTER = 10
EXAMPLES_PER_SECTION = 5

CHAPTER_EXPERIMENTS = {
    "01": {
        "experiment_dir": "experiments/01-mnist-cnn-experiments",
        "entrypoint": "train_mlp.py",
        "model_path": "experiments/01-mnist-cnn-experiments/mnist_experiments/models.py",
        "runner_path": "experiments/01-mnist-cnn-experiments/mnist_experiments/runner.py",
        "data_path": "experiments/01-mnist-cnn-experiments/mnist_experiments/data.py",
    },
    "02": {
        "experiment_dir": "experiments/01-mnist-cnn-experiments",
        "entrypoint": "train_cnn.py",
        "model_path": "experiments/01-mnist-cnn-experiments/mnist_experiments/models.py",
        "runner_path": "experiments/01-mnist-cnn-experiments/mnist_experiments/runner.py",
        "data_path": "experiments/01-mnist-cnn-experiments/mnist_experiments/data.py",
    },
    "03": {
        "experiment_dir": "experiments/02-cifar10-cnn-experiments",
        "entrypoint": "train_resnet.py",
        "model_path": "experiments/02-cifar10-cnn-experiments/cifar10_experiments/models.py",
        "runner_path": "experiments/02-cifar10-cnn-experiments/cifar10_experiments/runner.py",
        "data_path": "experiments/02-cifar10-cnn-experiments/cifar10_experiments/data.py",
    },
    "04": {
        "experiment_dir": "experiments/03-char-transformer-experiments",
        "entrypoint": "train_transformer.py",
        "model_path": "experiments/03-char-transformer-experiments/char_transformer_experiments/models.py",
        "runner_path": "experiments/03-char-transformer-experiments/char_transformer_experiments/runner.py",
        "data_path": "experiments/03-char-transformer-experiments/char_transformer_experiments/data.py",
    },
    "05": {
        "experiment_dir": "experiments/03-char-transformer-experiments",
        "entrypoint": "train_transformer.py",
        "model_path": "experiments/03-char-transformer-experiments/char_transformer_experiments/models.py",
        "runner_path": "experiments/03-char-transformer-experiments/char_transformer_experiments/runner.py",
        "data_path": "experiments/03-char-transformer-experiments/char_transformer_experiments/data.py",
    },
    "06": {
        "experiment_dir": "experiments/04-subword-gpt-experiments",
        "entrypoint": "train_gpt.py",
        "model_path": "experiments/04-subword-gpt-experiments/subword_gpt_experiments/models.py",
        "runner_path": "experiments/04-subword-gpt-experiments/subword_gpt_experiments/runner.py",
        "data_path": "experiments/04-subword-gpt-experiments/subword_gpt_experiments/tokenizer.py",
    },
}
GENERIC_SECTION_TITLES = (
    "关键结果",
    "本章目标",
    "继续阅读",
    "代码入口",
    "小结",
    "学完这章应掌握",
    "本章实验",
    "本章实验衔接",
)


@dataclass(slots=True)
class NoteSection:
    chapter_id: str
    chapter_title: str
    note_path: Path
    section_index: int
    section_title: str
    body: str
    chapter_intro: str
    previous_title: str
    next_title: str


def run_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    summary = build_dataset(config, overwrite=args.overwrite)

    split_counts = summary["split_counts"]
    print(f"Dataset: {config.dataset_path}")
    print(
        "Examples: "
        f"train={split_counts['train']} "
        f"val={split_counts['val']} "
        f"test={split_counts['test']}"
    )


def build_parser() -> argparse.ArgumentParser:
    default_config = create_default_config()
    parser = argparse.ArgumentParser(
        description="Build the notes-based instruction dataset for SFT.",
    )
    parser.add_argument("--notes-dir", default=str(default_config.notes_dir))
    parser.add_argument("--data-dir", default=str(default_config.data_dir))
    parser.add_argument(
        "--dataset-filename",
        default=default_config.dataset_filename,
    )
    parser.add_argument(
        "--dataset-summary-filename",
        default=default_config.dataset_summary_filename,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the dataset even if it already exists.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        notes_dir=Path(args.notes_dir),
        data_dir=Path(args.data_dir),
        dataset_filename=args.dataset_filename,
        dataset_summary_filename=args.dataset_summary_filename,
    )


def ensure_dataset(config: ExperimentConfig, overwrite: bool = False) -> dict[str, object]:
    if (
        config.dataset_path.exists()
        and config.dataset_summary_path.exists()
        and not overwrite
    ):
        return read_json(config.dataset_summary_path)
    return build_dataset(config, overwrite=overwrite)


def build_dataset(config: ExperimentConfig, overwrite: bool = False) -> dict[str, object]:
    if (
        config.dataset_path.exists()
        and config.dataset_summary_path.exists()
        and not overwrite
    ):
        return read_json(config.dataset_summary_path)

    chapters = load_note_sections(config.notes_dir)
    selected_sections = select_sections_for_dataset(chapters)

    records: list[dict[str, object]] = []
    chapter_summaries: list[dict[str, object]] = []

    for chapter_id in sorted(selected_sections):
        chapter_sections = split_sections_by_hash(selected_sections[chapter_id])
        chapter_counts = {"train": 0, "val": 0, "test": 0}
        selected_titles: list[str] = []
        for split_name, sections in chapter_sections.items():
            for section in sections:
                selected_titles.append(section.section_title)
                section_records = build_examples_for_section(section, split_name)
                chapter_counts[split_name] += len(section_records)
                records.extend(section_records)

        chapter_summaries.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": selected_sections[chapter_id][0].chapter_title,
                "note_path": str(selected_sections[chapter_id][0].note_path),
                "section_count": len(selected_sections[chapter_id]),
                "selected_sections": selected_titles,
                "split_counts": chapter_counts,
            }
        )

    records.sort(key=lambda item: str(item["id"]))

    split_counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        split_counts[str(record["split"])] += 1

    summary: dict[str, object] = {
        "dataset_path": str(config.dataset_path),
        "notes_dir": str(config.notes_dir),
        "total_examples": len(records),
        "split_counts": split_counts,
        "chapters": chapter_summaries,
        "sections_per_chapter": SECTIONS_PER_CHAPTER,
        "examples_per_section": EXAMPLES_PER_SECTION,
    }
    write_jsonl(config.dataset_path, records)
    write_json(config.dataset_summary_path, summary)
    return summary


def load_note_sections(notes_dir: Path) -> dict[str, list[NoteSection]]:
    sections_by_chapter: dict[str, list[NoteSection]] = {}
    for chapter_id in sorted(CHAPTER_EXPERIMENTS):
        matches = sorted(notes_dir.glob(f"{chapter_id}-*.md"))
        if not matches:
            raise FileNotFoundError(f"Missing note file for chapter {chapter_id}")
        sections_by_chapter[chapter_id] = parse_note_sections(matches[0], chapter_id)
    return sections_by_chapter


def parse_note_sections(note_path: Path, chapter_id: str) -> list[NoteSection]:
    lines = note_path.read_text(encoding="utf-8").splitlines()
    chapter_title = note_path.stem
    chapter_intro_lines: list[str] = []
    section_entries: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    in_code_fence = False

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue

        if line.startswith("# "):
            chapter_title = line[2:].strip()
            continue

        heading_match = re.match(r"^(#{2,4})\s+(.+)$", line)
        if heading_match:
            if current_title:
                section_entries.append((current_title, current_lines))
            current_title = heading_match.group(2).strip()
            current_lines = []
            continue

        cleaned_line = clean_markdown_line(line)
        if current_title:
            if cleaned_line:
                current_lines.append(cleaned_line)
        elif cleaned_line:
            chapter_intro_lines.append(cleaned_line)

    if current_title:
        section_entries.append((current_title, current_lines))

    chapter_intro = build_summary(
        " ".join(chapter_intro_lines),
        max_sentences=3,
        max_chars=220,
    )
    sections: list[NoteSection] = []
    for index, (section_title, body_lines) in enumerate(section_entries, start=1):
        body = normalize_text(" ".join(body_lines))
        if len(body) < 80:
            continue
        previous_title = section_entries[index - 2][0] if index > 1 else ""
        next_title = section_entries[index][0] if index < len(section_entries) else ""
        sections.append(
            NoteSection(
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                note_path=note_path,
                section_index=index,
                section_title=section_title,
                body=body,
                chapter_intro=chapter_intro,
                previous_title=previous_title,
                next_title=next_title,
            )
        )
    sections = augment_sections(sections, target_count=SECTIONS_PER_CHAPTER)
    if len(sections) < SECTIONS_PER_CHAPTER:
        raise ValueError(
            f"{note_path.name} only yielded {len(sections)} usable H2 sections."
        )
    return sections


def clean_markdown_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("<img") or stripped.startswith("</p>") or stripped.startswith("<p"):
        return ""
    if stripped.startswith("!"):
        return ""
    if re.fullmatch(r"\|?[-:\s|]+\|?", stripped):
        return ""

    stripped = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", stripped)
    stripped = re.sub(r"<[^>]+>", "", stripped)
    stripped = stripped.replace("`", "")
    stripped = re.sub(r"^\s*[-*+]\s+", "", stripped)
    stripped = re.sub(r"^\s*\d+\.\s+", "", stripped)
    stripped = stripped.replace("|", " ")
    return normalize_text(stripped)


def select_sections_for_dataset(
    sections_by_chapter: dict[str, list[NoteSection]]
) -> dict[str, list[NoteSection]]:
    selected: dict[str, list[NoteSection]] = {}
    for chapter_id, sections in sections_by_chapter.items():
        preferred_sections = [
            section for section in sections if not is_generic_section(section.section_title)
        ]
        candidate_sections = preferred_sections or sections
        candidate_sections = augment_sections(
            candidate_sections,
            target_count=SECTIONS_PER_CHAPTER,
        )
        selected[chapter_id] = pick_evenly_distributed_sections(
            candidate_sections,
            count=SECTIONS_PER_CHAPTER,
        )
    return selected


def is_generic_section(section_title: str) -> bool:
    return any(keyword in section_title for keyword in GENERIC_SECTION_TITLES)


def pick_evenly_distributed_sections(
    sections: list[NoteSection],
    count: int,
) -> list[NoteSection]:
    if len(sections) < count:
        raise ValueError(f"Need at least {count} sections, got {len(sections)}")
    if len(sections) == count:
        return sections

    indices: list[int] = []
    for step in range(count):
        candidate = round(step * (len(sections) - 1) / (count - 1))
        if candidate not in indices:
            indices.append(candidate)

    probe = 0
    while len(indices) < count:
        if probe not in indices:
            indices.append(probe)
        probe += 1
    indices.sort()
    return [sections[index] for index in indices[:count]]


def augment_sections(
    sections: list[NoteSection],
    *,
    target_count: int,
) -> list[NoteSection]:
    if len(sections) >= target_count:
        return sections

    augmented = list(sections)
    seen_titles = {section.section_title for section in sections}
    for section in sorted(sections, key=lambda item: len(item.body), reverse=True):
        sentence_chunks = chunk_section_sentences(section)
        for chunk_index, chunk_text in enumerate(sentence_chunks, start=1):
            synthetic_title = f"{section.section_title}（补充{chunk_index}）"
            if synthetic_title in seen_titles:
                continue
            augmented.append(
                NoteSection(
                    chapter_id=section.chapter_id,
                    chapter_title=section.chapter_title,
                    note_path=section.note_path,
                    section_index=section.section_index,
                    section_title=synthetic_title,
                    body=chunk_text,
                    chapter_intro=section.chapter_intro,
                    previous_title=section.section_title,
                    next_title=section.next_title,
                )
            )
            seen_titles.add(synthetic_title)
            if len(augmented) >= target_count:
                return augmented
    if augmented and len(augmented) < target_count and augmented[0].chapter_intro:
        intro_title = "本章主线概览"
        if intro_title not in seen_titles:
            seed = augmented[0]
            augmented.append(
                NoteSection(
                    chapter_id=seed.chapter_id,
                    chapter_title=seed.chapter_title,
                    note_path=seed.note_path,
                    section_index=0,
                    section_title=intro_title,
                    body=seed.chapter_intro,
                    chapter_intro=seed.chapter_intro,
                    previous_title="",
                    next_title=seed.section_title,
                )
            )
            seen_titles.add(intro_title)
            if len(augmented) >= target_count:
                return augmented
    synthetic_index = 1
    while augmented and len(augmented) < target_count:
        seed = augmented[(synthetic_index - 1) % len(augmented)]
        synthetic_title = f"{seed.section_title}（复盘{synthetic_index}）"
        if synthetic_title in seen_titles:
            synthetic_index += 1
            continue
        augmented.append(
            NoteSection(
                chapter_id=seed.chapter_id,
                chapter_title=seed.chapter_title,
                note_path=seed.note_path,
                section_index=seed.section_index,
                section_title=synthetic_title,
                body=build_summary(seed.body, max_sentences=3, max_chars=180),
                chapter_intro=seed.chapter_intro,
                previous_title=seed.previous_title,
                next_title=seed.next_title,
            )
        )
        seen_titles.add(synthetic_title)
        synthetic_index += 1
    return augmented


def chunk_section_sentences(section: NoteSection) -> list[str]:
    sentences = split_sentences(section.body)
    if len(sentences) < 4:
        return []

    chunk_size = 3 if len(sentences) >= 6 else 2
    chunks: list[str] = []
    for start in range(0, len(sentences), chunk_size):
        chunk = normalize_text(" ".join(sentences[start : start + chunk_size]))
        if len(chunk) >= 60:
            chunks.append(chunk)
    return chunks


def split_sections_by_hash(
    sections: Iterable[NoteSection],
) -> dict[str, list[NoteSection]]:
    ordered = sorted(
        sections,
        key=lambda item: stable_hash(f"{item.chapter_id}-{item.section_title}"),
    )
    return {
        "train": ordered[:8],
        "val": ordered[8:9],
        "test": ordered[9:10],
    }


def build_examples_for_section(
    section: NoteSection,
    split_name: str,
) -> list[dict[str, object]]:
    generators = (
        ("explain_core", build_explain_example),
        ("key_points", build_key_points_example),
        ("study_focus", build_study_focus_example),
        ("chapter_position", build_chapter_position_example),
        ("experiment_bridge", build_experiment_bridge_example),
    )

    records: list[dict[str, object]] = []
    for template_id, generator in generators:
        instruction, model_input, output = generator(section)
        section_slug = slugify(section.section_title, max_length=32)
        records.append(
            {
                "id": (
                    f"notes-assistant-{section.chapter_id}-"
                    f"{section_slug}-{template_id}"
                ),
                "instruction": instruction,
                "input": model_input,
                "output": output,
                "source_chapter": section.chapter_title,
                "source_section": section.section_title,
                "source_path": str(section.note_path),
                "template_id": template_id,
                "split": split_name,
            }
        )
    return records


def build_explain_example(section: NoteSection) -> tuple[str, str, str]:
    summary = build_summary(section.body, max_sentences=4, max_chars=300)
    output = (
        f"“{section.section_title}”这一节主要在讲：{summary} "
        f"把它放回《{section.chapter_title}》这章里看，它的作用是帮助你把本章主线"
        f"从概念直觉推进到更具体的实现或训练问题。"
    )
    return (
        f"请解释“{section.section_title}”的核心含义，并说明它在这一章里解决什么问题。",
        f"章节：{section.chapter_title}",
        output,
    )


def build_key_points_example(section: NoteSection) -> tuple[str, str, str]:
    points = build_key_points(section.body, limit=3)
    lines = [f"{index}. {point}" for index, point in enumerate(points, start=1)]
    output = "这部分可以先抓住下面 3 点：\n" + "\n".join(lines)
    return (
        f"总结《{section.chapter_title}》里“{section.section_title}”这部分的关键要点。",
        "请尽量用精炼的条目回答。",
        output,
    )


def build_study_focus_example(section: NoteSection) -> tuple[str, str, str]:
    focus_points = build_key_points(section.body, limit=2)
    short_summary = build_summary(section.body, max_sentences=2, max_chars=160)
    output = (
        f"复习这一节时，建议先盯住两件事：1. {focus_points[0]} 2. {focus_points[1]} "
        f"如果时间很紧，至少要记住：{short_summary}"
    )
    return (
        f"如果我要快速复习“{section.section_title}”，最值得优先抓住什么？",
        "请按学习重点而不是按百科定义来回答。",
        output,
    )


def build_chapter_position_example(
    section: NoteSection,
) -> tuple[str, str, str]:
    summary = build_summary(section.body, max_sentences=2, max_chars=180)
    lead_parts: list[str] = []
    if section.previous_title:
        lead_parts.append(f"它承接了前面的“{section.previous_title}”")
    else:
        lead_parts.append("它位于本章前半段的关键铺垫位置")
    if section.next_title:
        lead_parts.append(f"也为后面的“{section.next_title}”做准备")
    else:
        lead_parts.append("并负责把本章主线收束到更完整的理解上")

    output = (
        f"{'，'.join(lead_parts)}。具体来说，这一节主要强调：{summary} "
        f"所以它不是孤立知识点，而是本章学习顺序里的一个过渡节点。"
    )
    return (
        f"“{section.section_title}”在《{section.chapter_title}》这一章的学习顺序里处在什么位置？",
        "请回答它承接什么、又为后面什么内容做准备。",
        output,
    )


def build_experiment_bridge_example(
    section: NoteSection,
) -> tuple[str, str, str]:
    experiment = CHAPTER_EXPERIMENTS[section.chapter_id]
    summary = build_summary(section.body, max_sentences=2, max_chars=180)
    output = (
        f"如果要把这一节落到代码，建议先看 {experiment['experiment_dir']} 下的 "
        f"{experiment['entrypoint']} 跑通入口，再结合 {pick_support_path(section)} "
        f"理解这一节真正影响的是哪一段实现。对应到当前小节，最值得带着代码去验证的是：{summary}"
    )
    return (
        f"如果我想把“{section.section_title}”落到当前仓库里的实验代码，应该先看哪里？",
        "回答时请给出目录或文件入口，并说明为什么。",
        output,
    )


def pick_support_path(section: NoteSection) -> str:
    experiment = CHAPTER_EXPERIMENTS[section.chapter_id]
    title = section.section_title.lower()
    body = section.body.lower()
    if any(keyword in title or keyword in body for keyword in ("数据", "dataset", "padding", "token", "分词", "语料")):
        return experiment["data_path"]
    if any(
        keyword in title or keyword in body
        for keyword in ("训练", "优化", "loss", "采样", "生成", "评估")
    ):
        return experiment["runner_path"]
    return experiment["model_path"]


def build_summary(text: str, max_sentences: int, max_chars: int) -> str:
    sentences = split_sentences(text)
    selected: list[str] = []
    total_chars = 0
    for sentence in sentences:
        cleaned = normalize_text(sentence)
        if len(cleaned) < 12:
            continue
        if cleaned in selected:
            continue
        projected = total_chars + len(cleaned)
        if projected > max_chars and selected:
            break
        selected.append(cleaned)
        total_chars = projected
        if len(selected) >= max_sentences:
            break
    if not selected:
        return normalize_text(text)[:max_chars]
    return " ".join(selected)


def build_key_points(text: str, limit: int) -> list[str]:
    sentences = split_sentences(text)
    points: list[str] = []
    for sentence in sentences:
        cleaned = normalize_text(sentence)
        if len(cleaned) < 14:
            continue
        if cleaned in points:
            continue
        points.append(cleaned)
        if len(points) >= limit:
            break

    if not points:
        points.append(normalize_text(text)[:120])
    while len(points) < limit:
        points.append(points[-1])
    return points[:limit]


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\n", " ")
    chunks = re.split(r"(?<=[。！？!?；;])\s+|(?<=\.)\s+", normalized)
    return [chunk.strip(" -") for chunk in chunks if chunk.strip(" -")]
