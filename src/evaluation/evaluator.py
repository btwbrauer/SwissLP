"""
Evaluation utilities for model evaluation.

Provides Evaluator classes for evaluating trained models with
comprehensive metrics and result reporting.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from .metrics import ClassificationMetrics


class Evaluator(ABC):
    """Base evaluator class for model evaluation."""

    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        data_loader: DataLoader,
        class_names: list[str] | None = None,
    ):
        """
        Initialize evaluator.

        Args:
            config: Configuration object
            model: Model to evaluate
            data_loader: Data loader for evaluation
            class_names: Optional list of class names for metrics
        """
        self.config = config
        self.model = model
        self.data_loader = data_loader

        # Setup device
        self.device = (
            torch.device(config.model.device)
            if config.model.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        # Setup metrics
        num_labels = config.model.num_labels or 8
        self.metrics = ClassificationMetrics(num_labels, class_names=class_names)

    @abstractmethod
    def _eval_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Perform a single evaluation step."""
        pass

    def evaluate(self) -> dict[str, Any]:
        """
        Evaluate the model on the dataset.

        Returns:
            Dictionary of evaluation results
        """
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Evaluating"):
                eval_output = self._eval_step(batch)
                total_loss += eval_output.get("loss", 0.0)
                num_batches += 1

                self.metrics.update(eval_output["predictions"], eval_output["labels"])

        metrics = self.metrics.compute()

        # Only add loss if it was computed
        if num_batches > 0 and total_loss > 0:
            metrics["loss"] = total_loss / num_batches

        return metrics

    def save_results(self, results: dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)


class TextEvaluator(Evaluator):
    """Evaluator for text classification models."""

    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        tokenizer: Any,
        data_loader: DataLoader,
        compute_loss: bool = True,
        class_names: list[str] | None = None,
    ):
        """
        Initialize text evaluator.

        Args:
            config: Configuration object
            model: Text classification model
            tokenizer: Tokenizer for text preprocessing
            data_loader: Data loader for evaluation
            compute_loss: Whether to compute loss (default: True)
            class_names: Optional list of class names for metrics
        """
        super().__init__(config, model, data_loader, class_names=class_names)
        self.tokenizer = tokenizer
        self.compute_loss = compute_loss
        if compute_loss:
            self.criterion = torch.nn.CrossEntropyLoss()

    def _eval_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Perform a single evaluation step for text."""
        texts = batch["text"]
        labels = batch["label"].to(self.device)

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.data.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Get predictions
        predictions = logits.argmax(dim=-1)

        result = {"predictions": predictions, "labels": labels}

        # Compute loss if requested
        if self.compute_loss:
            loss = self.criterion(logits, labels)
            result["loss"] = loss.item()

        return result


class SpeechEvaluator(Evaluator):
    """Evaluator for speech classification models."""

    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        processor: Any,
        data_loader: DataLoader,
        compute_loss: bool = True,
        class_names: list[str] | None = None,
    ):
        """
        Initialize speech evaluator.

        Args:
            config: Configuration object
            model: Speech classification model
            processor: Audio processor for preprocessing
            data_loader: Data loader for evaluation
            compute_loss: Whether to compute loss (default: True)
            class_names: Optional list of class names for metrics
        """
        super().__init__(config, model, data_loader, class_names=class_names)
        self.processor = processor
        self.compute_loss = compute_loss
        if compute_loss:
            self.criterion = torch.nn.CrossEntropyLoss()

    def _eval_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Perform a single evaluation step for speech."""
        audio = batch["audio"].to(self.device)
        labels = batch["label"].to(self.device)

        # Process audio - audio is already padded from collate function
        # Convert to numpy for processor if needed
        if isinstance(audio, torch.Tensor):
            # Processor expects [batch, samples] or list of 1D arrays
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Process with feature extractor/processor
        inputs = self.processor(
            audio_np,
            sampling_rate=self.config.data.audio_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Get predictions
        predictions = logits.argmax(dim=-1)

        result = {"predictions": predictions, "labels": labels}

        # Compute loss if requested
        if self.compute_loss:
            loss = self.criterion(logits, labels)
            result["loss"] = loss.item()

        return result
