from __future__ import annotations

from pathlib import Path

from .utils import ensure_parent_dir


def save_loss_curve(
    log_history: list[dict[str, object]],
    destination: Path,
) -> tuple[bool, str]:
    train_points = [
        (entry.get("step"), entry.get("loss"))
        for entry in log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    eval_points = [
        (entry.get("step"), entry.get("eval_loss"))
        for entry in log_history
        if "eval_loss" in entry
    ]
    if not train_points and not eval_points:
        return False, "No Trainer log history was available."

    try:
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        return False, f"matplotlib unavailable: {error}"

    ensure_parent_dir(destination)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if train_points:
        axis.plot(
            [point[0] for point in train_points],
            [point[1] for point in train_points],
            label="train_loss",
            linewidth=2,
        )
    if eval_points:
        axis.plot(
            [point[0] for point in eval_points],
            [point[1] for point in eval_points],
            label="eval_loss",
            linewidth=2,
        )
    axis.set_title("SFT Loss Curve")
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True, "saved"
