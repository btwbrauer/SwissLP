# 🇨🇭 Swiss Language Processing (SwissLP)

A minimal, best-practice repository for Text and Speech Processing using Swiss German and German language models.

## 📚 Course: Essentials in Text and Speech Processing

This project provides a clean, reproducible development environment for Swiss German dialect classification.

## 📊 Dataset

This project uses the **SwissDial** dataset for Swiss German dialect classification tasks.

## 🚀 Features

### Text Models

- **SwissBERT**: BERT model fine-tuned on Swiss German data
- **German BERT**: Standard BERT model for German language
- **XLM-RoBERTa**: Cross-lingual model supporting 100+ languages
- **fastText**: Pre-trained word embeddings (located in `crawl-300d-2M-subword/`)

### Speech Models

- **Wav2Vec2**: Facebook's self-supervised speech representation
- **AST**: Audio Spectrogram Transformer
- **Whisper**: OpenAI's robust speech recognition model

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/btwbrauer/SwissLP.git
cd SwissLP
```

1. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install dependencies:**

```bash
pip install -e .
```

1. **Verify installation:**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
pytest tests/ -v
```

**Note**: The fastText embeddings are located in the `crawl-300d-2M-subword/` directory. This directory is gitignored due to file size.

## 📁 Project Structure

```text
SwissLP/
├── pyproject.toml         # Python dependencies and project configuration
├── README.md              # This file
├── src/
│   ├── models/            # Model loaders
│   │   ├── text_models.py    # Text model loaders
│   │   └── speech_models.py # Speech model loaders
│   ├── evaluation/        # Evaluation utilities
│   │   ├── evaluator.py      # Evaluator classes
│   │   └── metrics.py        # Metrics computation
│   ├── config/            # Configuration management
│   │   └── config.py         # Config classes and YAML loading
│   ├── training/          # Training utilities
│   │   ├── trainer.py        # Trainer classes
│   │   ├── hyperparameter_tuning.py # Optuna hyperparameter tuning
│   │   └── optimization.py  # Model optimization workflow
│   └── utils/             # Utility functions
│       ├── dataset.py        # Dataset loading and preparation
│       └── mlflow_utils.py   # MLflow integration
├── scripts/               # Executable scripts
│   └── optimize_models.py   # Hyperparameter optimization script
├── configs/               # Configuration files
│   ├── swissbert.yaml
│   ├── german_bert.yaml
│   ├── xlm_roberta.yaml
│   └── wav2vec2.yaml
└── tests/                 # Test suite
```

## 🎓 Training Models

Fine-tune models on the Swiss German dialect dataset using hyperparameter optimization:

```bash
python scripts/optimize_models.py --models swissbert --n-trials 20
```

```bash
python scripts/optimize_models.py --models german_bert --n-trials 20
```

```bash
python scripts/optimize_models.py --models xlm_roberta --n-trials 20
```

```bash
python scripts/optimize_models.py --models swissbert german_bert xlm_roberta --n-trials 20
```

```bash
python scripts/optimize_models.py --models swissbert --n-trials 30 --timeout 7200
```

```bash
python scripts/optimize_models.py --models swissbert --n-trials 20 --skip-cleanup
```

The optimization script will:

- Run hyperparameter tuning using Optuna
- Automatically log all trials to MLflow
- Keep only the best model (unless `--skip-cleanup` is used)
- Save optimized config files to `configs/`

## 🔬 Evaluating and Comparing Models

After training, evaluate your fine-tuned models. Evaluation is done programmatically using the evaluation utilities in `src/evaluation/`. Load trained models from the `outputs/` directory and evaluate them on the test dataset.

The evaluation module provides:

- `TextEvaluator` and `SpeechEvaluator` classes
- Comprehensive metrics (accuracy, precision, recall, F1-score)
- Per-class metrics computation
- Results saving to JSON/CSV

## 📊 MLflow Experiment Tracking

All training runs are automatically logged to MLflow for experiment tracking and comparison.

### MLflow Configuration

By default, training scripts log to <http://localhost:5000> (remote MLflow server).

**To use a different MLflow server**, set the `MLFLOW_TRACKING_URI` environment variable:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

```bash
export MLFLOW_TRACKING_URI="./mlruns"
```

### What Gets Logged

MLflow automatically tracks:

- **Parameters**: Model configuration, hyperparameters, data splits
- **Metrics**: Training loss, validation metrics (accuracy, F1, etc.)
- **Artifacts**: Trained models, checkpoints, config files
- **Experiment metadata**: Run name, timestamp, experiment name

## ⚙️ Configuration

Model training is configured using YAML files in the `configs/` directory.

Key configuration sections:

- **model**: Model name, number of labels, device
- **data**: Data path, dialects, train/val/test splits, batch_size
- **training**: Learning rate, batch size, epochs, early stopping, etc.
- **experiment_name**: Name for organizing outputs

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 🔬 Development

### Code Quality

Format code:

```bash
ruff format src/ tests/
```

Lint code:

```bash
ruff check src/ tests/
```

## 📦 Dependencies

See `pyproject.toml` for the complete list of dependencies with exact versions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Format code: `ruff format src/ tests/`
6. Submit a pull request

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and model hub
- **PyTorch Team**: For the deep learning framework
- **Course Instructors**: For guidance and support

## 📚 Resources

- [Hugging Face Models](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SwissBERT Paper](https://aclanthology.org/2023.swisstext-1.3/)
