"""
Tests for configuration management.

Tests the Config class and configuration loading/saving.
"""

import tempfile
from pathlib import Path

from src.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(model_name="test-model", num_labels=8)
        assert config.model_name == "test-model"
        assert config.num_labels == 8


class TestDataConfig:
    """Tests for DataConfig."""

    def test_data_config_creation(self):
        """Test DataConfig creation."""
        config = DataConfig(data_path="./data/test.json")
        assert config.data_path == "./data/test.json"
        assert config.train_ratio == 0.8
        assert config.batch_size == 16  # batch_size is now in DataConfig


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_training_config_creation(self):
        """Test TrainingConfig creation with defaults."""
        config = TrainingConfig()
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.batch_size == 16
        assert config.warmup_steps == 100
        assert config.weight_decay == 0.01
        assert config.max_grad_norm == 1.0
        assert config.gradient_accumulation_steps == 1
        assert config.save_steps == 500
        assert config.eval_steps == 500
        assert config.logging_steps == 100
        assert config.output_dir == "./outputs"
        assert config.seed == 42
        assert config.fp16 is False
        assert config.lr_scheduler_type == "linear"
        assert config.early_stopping_patience is None
        assert config.early_stopping_threshold == 0.0

    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=32,
            fp16=True,
            lr_scheduler_type="cosine",
            early_stopping_patience=5,
            early_stopping_threshold=0.01,
        )
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 5
        assert config.batch_size == 32
        assert config.fp16 is True
        assert config.lr_scheduler_type == "cosine"
        assert config.early_stopping_patience == 5
        assert config.early_stopping_threshold == 0.01


class TestConfig:
    """Tests for Config."""

    def test_config_creation(self):
        """Test Config creation."""
        config = Config(
            model=ModelConfig(model_name="test-model"),
            data=DataConfig(data_path="./data/test.json"),
        )
        assert config.model.model_name == "test-model"
        assert config.experiment_name == "default"
        assert config.training is None

    def test_config_creation_with_training(self):
        """Test Config creation with training config."""
        training_config = TrainingConfig(learning_rate=1e-4, num_epochs=5)
        config = Config(
            model=ModelConfig(model_name="test-model"),
            data=DataConfig(data_path="./data/test.json"),
            training=training_config,
        )
        assert config.training is not None
        assert config.training.learning_rate == 1e-4
        assert config.training.num_epochs == 5

    def test_config_to_dict(self):
        """Test Config to_dict conversion."""
        config = get_default_config()
        config_dict = config.to_dict()

        assert "model" in config_dict
        assert "data" in config_dict
        assert "experiment_name" in config_dict
        # training is optional, so it may or may not be in the dict

    def test_config_to_dict_with_training(self):
        """Test Config to_dict with training config."""
        training_config = TrainingConfig(learning_rate=1e-4)
        config = Config(
            model=ModelConfig(model_name="test-model"),
            data=DataConfig(data_path="./data/test.json"),
            training=training_config,
        )
        config_dict = config.to_dict()

        assert "training" in config_dict
        assert config_dict["training"]["learning_rate"] == 1e-4
        assert config_dict["training"]["num_epochs"] == 3  # default value

    def test_config_from_dict(self):
        """Test Config from_dict creation."""
        config_dict = {
            "model": {"model_name": "test-model", "num_labels": 8},
            "data": {"data_path": "./data/test.json", "batch_size": 32},
            "experiment_name": "test_exp",
            "task_type": "text",
        }

        config = Config.from_dict(config_dict)
        assert config.model.model_name == "test-model"
        assert config.model.num_labels == 8
        assert config.data.batch_size == 32
        assert config.experiment_name == "test_exp"
        assert config.training is None

    def test_config_from_dict_with_training(self):
        """Test Config from_dict with training config."""
        config_dict = {
            "model": {"model_name": "test-model", "num_labels": 8},
            "data": {"data_path": "./data/test.json"},
            "experiment_name": "test_exp",
            "task_type": "text",
            "training": {
                "learning_rate": 1e-4,
                "num_epochs": 5,
                "batch_size": 32,
            },
        }

        config = Config.from_dict(config_dict)
        assert config.training is not None
        assert config.training.learning_rate == 1e-4
        assert config.training.num_epochs == 5
        assert config.training.batch_size == 32

    def test_config_save_and_load(self):
        """Test Config save and load."""
        config = get_default_config()
        config.experiment_name = "test_experiment"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)
            loaded_config = Config.load(temp_path)

            assert loaded_config.experiment_name == "test_experiment"
            assert loaded_config.model.model_name == config.model.model_name
        finally:
            Path(temp_path).unlink()

    def test_config_save_and_load_with_training(self):
        """Test Config save and load with training config."""
        training_config = TrainingConfig(learning_rate=1e-4, num_epochs=5)
        config = Config(
            model=ModelConfig(model_name="test-model"),
            data=DataConfig(data_path="./data/test.json"),
            training=training_config,
            experiment_name="test_experiment",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)
            loaded_config = Config.load(temp_path)

            assert loaded_config.experiment_name == "test_experiment"
            assert loaded_config.training is not None
            assert loaded_config.training.learning_rate == 1e-4
            assert loaded_config.training.num_epochs == 5
        finally:
            Path(temp_path).unlink()

    def test_get_default_config(self):
        """Test get_default_config function."""
        config = get_default_config()
        assert isinstance(config, Config)
        assert config.model.model_name is not None
        assert config.data.data_path is not None
