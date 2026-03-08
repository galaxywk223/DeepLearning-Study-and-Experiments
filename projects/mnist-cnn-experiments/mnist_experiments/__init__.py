from .config import ExperimentConfig, create_default_config

__all__ = [
    "CNNClassifier",
    "ExperimentConfig",
    "MLPClassifier",
    "build_model",
    "create_default_config",
    "run_experiment",
]


def __getattr__(name: str):
    if name in {"CNNClassifier", "MLPClassifier", "build_model"}:
        from . import models

        return getattr(models, name)
    if name == "run_experiment":
        from .runner import run_experiment

        return run_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
