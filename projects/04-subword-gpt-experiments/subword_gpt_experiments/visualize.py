from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def save_loss_curve_in_subprocess(
    history: list[dict[str, float | int]],
    destination: Path,
    *,
    show_plot: bool,
) -> tuple[bool, str]:
    payload = {
        "history": history,
        "destination": str(destination),
        "show_plot": show_plot,
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        delete=False,
    ) as temp_file:
        json.dump(payload, temp_file, ensure_ascii=False)
        temp_path = Path(temp_file.name)

    script = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path

        payload_path = Path(sys.argv[1])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        history = payload["history"]
        destination = Path(payload["destination"])
        show_plot = payload["show_plot"]

        import matplotlib.pyplot as plt

        epochs = [entry["epoch"] for entry in history]
        train_losses = [entry["train_eval_loss"] for entry in history]
        val_losses = [entry["val_loss"] for entry in history]

        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(epochs, train_losses, label="Train loss", marker="o")
        axis.plot(epochs, val_losses, label="Val loss", marker="o")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.set_title("Subword GPT Loss Curve")
        axis.grid(alpha=0.2)
        axis.legend()
        fig.tight_layout()
        fig.savefig(destination, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)
        """
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(temp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if result.returncode == 0:
        return True, "saved"

    stderr = result.stderr.strip()
    if stderr:
        return False, stderr
    stdout = result.stdout.strip()
    if stdout:
        return False, stdout
    return False, "Unknown plotting error."

