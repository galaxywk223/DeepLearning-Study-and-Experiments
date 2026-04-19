from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from .config import (
    TEMPLATE_GROUPS,
    ExperimentConfig,
    create_default_config,
)
from .utils import ensure_dir, set_seed, summarize_trainable_parameters, write_json, write_text


def run_cli() -> None:
    # CLI 入口只负责参数解析与配置组装，训练细节集中在 run_training()。
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_training(config, overwrite_dataset=args.rebuild_dataset)


def build_parser() -> argparse.ArgumentParser:
    default_config = create_default_config()
    parser = argparse.ArgumentParser(
        description="Run LoRA/QLoRA supervised fine-tuning for the notes assistant.",
    )
    parser.add_argument(
        "--experiment-name",
        default=default_config.experiment_name,
        help="Subdirectory name under the output directory.",
    )
    parser.add_argument("--notes-dir", default=str(default_config.notes_dir))
    parser.add_argument("--data-dir", default=str(default_config.data_dir))
    parser.add_argument("--output-dir", default=str(default_config.output_dir))
    parser.add_argument(
        "--dataset-filename",
        default=default_config.dataset_filename,
    )
    parser.add_argument(
        "--dataset-summary-filename",
        default=default_config.dataset_summary_filename,
    )
    parser.add_argument("--model-id", default=default_config.model_id)
    parser.add_argument(
        "--system-prompt",
        default=default_config.system_prompt,
    )
    parser.add_argument(
        "--template-group",
        choices=tuple(TEMPLATE_GROUPS),
        default=default_config.template_group,
        help="Select which training/validation question templates are included.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=default_config.max_seq_length,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_config.per_device_train_batch_size,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=default_config.per_device_eval_batch_size,
    )
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
        "--weight-decay",
        type=float,
        default=default_config.weight_decay,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=default_config.warmup_ratio,
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=default_config.epochs,
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=default_config.max_train_samples,
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=default_config.max_eval_samples,
    )
    parser.add_argument("--lora-r", type=int, default=default_config.lora_r)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=default_config.lora_alpha,
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=default_config.lora_dropout,
    )
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument("--device-map", default=default_config.device_map)
    parser.add_argument(
        "--quantization-mode",
        choices=("4bit", "none"),
        default=default_config.quantization_mode,
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=default_config.logging_steps,
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=default_config.save_total_limit,
    )
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
    parser.add_argument("--top-p", type=float, default=default_config.top_p)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=default_config.repetition_penalty,
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the local JSONL dataset before training.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny dataset subset and one epoch for a quick smoke run.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    # 命令行参数在这里被收敛为统一配置对象，便于训练、评测和推理复用同一套配置。
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        notes_dir=Path(args.notes_dir),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        dataset_filename=args.dataset_filename,
        dataset_summary_filename=args.dataset_summary_filename,
        model_id=args.model_id,
        system_prompt=args.system_prompt,
        template_group=args.template_group,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        device_map=args.device_map,
        quantization_mode=args.quantization_mode,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    if args.smoke:
        # 冒烟模式只验证链路是否可运行，因此显式压缩样本量、轮数和日志间隔。
        config.experiment_name = f"{config.experiment_name}-smoke"
        config.max_train_samples = 16
        config.max_eval_samples = 8
        config.epochs = 1.0
        config.logging_steps = 1
    return config


def run_training(
    config: ExperimentConfig,
    *,
    overwrite_dataset: bool = False,
) -> dict[str, object]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from .data import SupervisedDataCollator, compose_user_message, load_sft_splits
    from .inference import (
        build_quantization_config,
        generate_answer,
        load_tokenizer,
        resolve_compute_dtype,
        resolve_device_map,
    )
    from .visualize import save_loss_curve

    set_seed(config.seed)
    run_dir = ensure_dir(config.run_dir)
    tokenizer = load_tokenizer(config.model_id)
    # 训练阶段右侧补齐，保证 assistant 末尾标签与因果语言模型的对齐方式一致。
    tokenizer.padding_side = "right"

    # 数据集加载阶段会确保本地 JSONL 已生成，并提前完成 train/val/test 划分与模板过滤。
    data_bundle = load_sft_splits(
        config,
        tokenizer,
        overwrite_dataset=overwrite_dataset,
    )
    collator = SupervisedDataCollator(tokenizer.pad_token_id)

    # 量化配置和 device_map 先被解析为 from_pretrained() 可直接接受的参数。
    quantization_config = build_quantization_config(config.quantization_mode)
    resolved_device_map = resolve_device_map(config.device_map)
    load_kwargs: dict[str, object] = {"trust_remote_code": False}
    if resolved_device_map is not None:
        load_kwargs["device_map"] = resolved_device_map
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["dtype"] = resolve_compute_dtype()

    try:
        # 这里先加载原始底座模型，LoRA adapter 还没有注入。
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            **load_kwargs,
        )
    except TypeError:
        if "dtype" not in load_kwargs:
            raise
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            **fallback_kwargs,
        )
    if quantization_config is not None:
        # QLoRA 路径下，底座模型先以低比特形式加载，再转换为适合 k-bit 训练的状态。
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    elif config.gradient_checkpointing:
        # 非量化路径只需显式开启 gradient checkpointing 以换取更低显存占用。
        model.gradient_checkpointing_enable()

    # LoRA 配置只描述 adapter 的形状和挂载目标，不直接触发训练。
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )
    # get_peft_model() 会在目标线性层上注入 LoRA adapter，并返回可训练的包装模型。
    model = get_peft_model(
        model,
        lora_config,
        autocast_adapter_dtype=False,
    )
    # 训练阶段禁用 KV cache，避免与梯度计算产生冲突。
    model.config.use_cache = False

    trainable_params, total_params = summarize_trainable_parameters(model)
    print(f"Run directory: {run_dir}")
    print(f"Dataset: {config.dataset_path}")
    print(
        f"Template group: {config.template_group} "
        f"({', '.join(config.selected_template_ids)})"
    )
    print(
        "Split sizes: "
        f"train={data_bundle.raw_counts['train']} "
        f"val={data_bundle.raw_counts['val']} "
        f"test={data_bundle.raw_counts['test']}"
    )
    print(
        "Template counts: "
        f"train={data_bundle.template_counts['train']} "
        f"val={data_bundle.template_counts['val']} "
        f"test={data_bundle.template_counts['test']}"
    )
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / max(total_params, 1):.2%})"
    )

    bf16_enabled = (
        config.quantization_mode != "none"
        and str(resolve_compute_dtype()) == "torch.bfloat16"
    )
    fp16_enabled = config.quantization_mode != "none" and not bf16_enabled

    # 训练参数按 Transformers 版本兼容地组装，避免旧版/新版字段名差异导致报错。
    training_kwargs: dict[str, object] = {
        "output_dir": str(run_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "num_train_epochs": config.epochs,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "gradient_checkpointing": config.gradient_checkpointing,
        "fp16": fp16_enabled,
        "bf16": bf16_enabled,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "seed": config.seed,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "overwrite_output_dir" in signature.parameters:
        training_kwargs["overwrite_output_dir"] = True
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    if "do_train" in signature.parameters:
        training_kwargs["do_train"] = True
    if "do_eval" in signature.parameters:
        training_kwargs["do_eval"] = True

    training_args = TrainingArguments(**training_kwargs)

    # Trainer 只关心三类核心对象：模型、数据集和 collator；LoRA 逻辑已在模型内部生效。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_bundle.train_dataset,
        eval_dataset=data_bundle.val_dataset,
        data_collator=collator,
    )

    # train() 更新的是 LoRA adapter 参数；冻结的底座权重不会参与梯度更新。
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    # 推理和采样阶段需要恢复 cache，以便后续生成更高效。
    trainer.model.config.use_cache = True

    ensure_dir(config.adapter_dir)
    # 保存的是 adapter 权重和 tokenizer，而不是一整份重新导出的底座模型。
    trainer.model.save_pretrained(config.adapter_dir)
    tokenizer.save_pretrained(config.adapter_dir)

    # 训练结束后补充导出可视化和样例，便于快速检查收敛情况与回答风格。
    loss_curve_saved, loss_curve_message = save_loss_curve(
        trainer.state.log_history,
        config.loss_curve_path,
    )
    samples_markdown = render_sample_markdown(
        trainer.model,
        tokenizer,
        config=config,
        records=data_bundle.test_records[:5],
    )
    write_text(config.sample_path, samples_markdown)

    metrics: dict[str, object] = {
        "config": config.to_dict(),
        "dataset_path": str(config.dataset_path),
        "split_counts": data_bundle.raw_counts,
        "template_group": config.template_group,
        "selected_template_ids": list(config.selected_template_ids),
        "split_template_counts": data_bundle.template_counts,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "loss_curve_saved": loss_curve_saved,
        "loss_curve_message": loss_curve_message,
        "artifacts": {
            "adapter_dir": str(config.adapter_dir),
            "config": str(config.config_path),
            "metrics": str(config.metrics_path),
            "samples": str(config.sample_path),
            "loss_curve": str(config.loss_curve_path),
        },
    }
    write_json(config.config_path, config.to_dict())
    write_json(config.metrics_path, metrics)
    return metrics


def render_sample_markdown(
    model,
    tokenizer,
    *,
    config: ExperimentConfig,
    records: list[dict[str, object]],
) -> str:
    from .data import compose_user_message
    from .inference import generate_answer

    lines = ["# Notes Assistant Sample Generations", ""]
    for index, record in enumerate(records, start=1):
        # 样例生成复用与正式评测一致的 prompt 构造逻辑，避免展示样例与训练设定脱节。
        question = compose_user_message(record)
        prediction = generate_answer(
            model,
            tokenizer,
            system_prompt=config.system_prompt,
            user_message=question,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )
        lines.extend(
            [
                f"## Sample {index}",
                "",
                f"**Source**: {record['source_chapter']} / {record['source_section']}",
                "",
                f"**Template**: {record['template_id']}",
                "",
                "**Question**",
                "",
                question,
                "",
                "**Reference**",
                "",
                str(record["output"]),
                "",
                "**Model Output**",
                "",
                prediction,
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"
