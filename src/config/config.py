"""
Configuration management for model inference and evaluation.

Provides configuration classes and utilities for managing model settings
and data loading in a reproducible manner.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for model loading and initialization."""

    model_name: str
    num_labels: int | None = None
    device: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_path: str
    dialects: list[str] = field(
        default_factory=lambda: [
            "ch_de",
            "ch_lu",
            "ch_be",
            "ch_zh",
            "ch_vs",
            "ch_bs",
            "ch_gr",
            "ch_sg",
        ]
    )
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_length: int = 512
    audio_sample_rate: int = 16000
    audio_max_duration: float | None = None
    batch_size: int = 16  # For inference/evaluation


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./outputs"
    seed: int = 42
    fp16: bool = False
    lr_scheduler_type: str = "linear"  # "linear", "cosine", "constant", "polynomial"
    early_stopping_patience: int | None = (
        None  # None = disabled, int = number of eval steps to wait
    )
    early_stopping_threshold: float = 0.0  # Minimum improvement required


@dataclass
class Config:
    """Main configuration class for model inference, training, and evaluation."""

    model: ModelConfig
    data: DataConfig
    experiment_name: str = "default"
    task_type: str = "text"  # "text" or "speech"
    training: TrainingConfig | None = None

    @classmethod
    def _normalize_numeric_value(cls, value: Any, target_type: type[float] | type[int]) -> Any:
        """Normalize numeric values from YAML (handles string representations)."""
        if value is None or isinstance(value, target_type):
            return value
        try:
            return target_type(float(value))
        except (ValueError, TypeError):
            return value

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        training_dict = config_dict.get("training")
        if training_dict:
            training_dict = training_dict.copy()
            float_keys = [
                "learning_rate",
                "weight_decay",
                "max_grad_norm",
                "early_stopping_threshold",
            ]
            int_keys = [
                "num_epochs",
                "batch_size",
                "warmup_steps",
                "gradient_accumulation_steps",
                "save_steps",
                "eval_steps",
                "logging_steps",
                "seed",
                "early_stopping_patience",
            ]
            for key in float_keys:
                if key in training_dict:
                    training_dict[key] = cls._normalize_numeric_value(training_dict[key], float)
            for key in int_keys:
                if key in training_dict:
                    training_dict[key] = cls._normalize_numeric_value(training_dict[key], int)
            if "fp16" in training_dict and isinstance(training_dict["fp16"], str):
                training_dict["fp16"] = training_dict["fp16"].lower() in ("true", "1", "yes")

        training = TrainingConfig(**training_dict) if training_dict else None

        data_dict = config_dict.get("data", {}).copy()
        for key in ["train_ratio", "val_ratio", "test_ratio", "audio_max_duration"]:
            if key in data_dict:
                data_dict[key] = cls._normalize_numeric_value(data_dict[key], float)
        for key in ["max_length", "audio_sample_rate", "batch_size"]:
            if key in data_dict:
                data_dict[key] = cls._normalize_numeric_value(data_dict[key], int)

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**data_dict),
            experiment_name=config_dict.get("experiment_name", "default"),
            task_type=config_dict.get("task_type", "text"),
            training=training,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Config to dictionary."""
        result = {
            "model": {
                "model_name": self.model.model_name,
                "num_labels": self.model.num_labels,
                "device": self.model.device,
                "model_kwargs": self.model.model_kwargs,
            },
            "data": {
                "data_path": self.data.data_path,
                "dialects": self.data.dialects,
                "train_ratio": self.data.train_ratio,
                "val_ratio": self.data.val_ratio,
                "test_ratio": self.data.test_ratio,
                "max_length": self.data.max_length,
                "audio_sample_rate": self.data.audio_sample_rate,
                "audio_max_duration": self.data.audio_max_duration,
                "batch_size": self.data.batch_size,
            },
            "experiment_name": self.experiment_name,
            "task_type": self.task_type,
        }

        if self.training is not None:
            result["training"] = {
                "learning_rate": self.training.learning_rate,
                "num_epochs": self.training.num_epochs,
                "batch_size": self.training.batch_size,
                "warmup_steps": self.training.warmup_steps,
                "weight_decay": self.training.weight_decay,
                "max_grad_norm": self.training.max_grad_norm,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "save_steps": self.training.save_steps,
                "eval_steps": self.training.eval_steps,
                "logging_steps": self.training.logging_steps,
                "output_dir": self.training.output_dir,
                "seed": self.training.seed,
                "fp16": self.training.fp16,
                "lr_scheduler_type": self.training.lr_scheduler_type,
                "early_stopping_patience": self.training.early_stopping_patience,
                "early_stopping_threshold": self.training.early_stopping_threshold,
            }

        return result

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with Path(path).open() as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from file or return default."""
    return get_default_config() if config_path is None else Config.load(config_path)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config(
        model=ModelConfig(model_name="ZurichNLP/swissbert", num_labels=8),
        data=DataConfig(data_path="./data/sentences_ch_de_transcribed.json"),
        experiment_name="default",
        task_type="text",
    )
