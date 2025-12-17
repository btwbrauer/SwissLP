# 🇨🇭 Swiss Language Processing (SwissLP)

A minimal, best-practice repository for Text and Speech Processing using Swiss German and German language models.

## 📚 Course: Essentials in Text and Speech Processing

This project provides a clean, reproducible development environment using Nix flakes for cross-platform compatibility.

## 🚀 Features

### Text Models
- **SwissBERT**: BERT model fine-tuned on Swiss German data
- **German BERT**: Standard BERT model for German language
- **XLM-RoBERTa**: Cross-lingual model supporting 100+ languages
- **ByT5**: Byte-level T5 model for multilingual tasks

### Speech Models
- **Wav2Vec2**: Facebook's self-supervised speech representation
- **AST**: Audio Spectrogram Transformer

### Hardware Support
- **NVIDIA GPUs**: Automatic CUDA support
- **AMD GPUs**: Automatic ROCm support
- **Apple Silicon**: MPS acceleration on macOS
- **CPU**: Fallback for all systems

## 🛠️ Installation

This repository is focused on running notebooks for comparing how the different AI models handle different Swiss German dialects. The simplest way to get started is with a Python virtual environment and pip.

Prerequisites:
- Python 3.10 or newer
- Optional: GPU with CUDA/ROCm or Apple Silicon (MPS) if you want hardware acceleration

1) Clone the repository
```bash
git clone https://github.com/btwbrauer/SwissLP.git
cd SwissLP
```

2) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

3) Install PyTorch
- Follow the official selector for your OS/accelerator: https://pytorch.org/get-started/locally/
- Example (CPU-only, pip):
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4) Install the remaining Python packages used in the notebooks/utilities
```bash
pip install --upgrade \
  transformers datasets sentencepiece accelerate \
  librosa soundfile \
  jupyter matplotlib pandas scikit-learn tqdm \
  fasttext
```

Notes
- If `torchaudio` did not install together with PyTorch for your platform, install a matching wheel from the PyTorch index (see step 3 link).
- For Apple Silicon, modern PyTorch wheels support MPS by default; no extra steps usually required.

## 📁 Project Structure

```
SwissLP/
├── README.md
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Audio/text loading utilities
│   │   └── prepare_dataset.py    # Helpers to create splits/tables
│   └── notebooks/
│       ├── AST.ipynb
│       ├── GeneralGermanBERT.ipynb
│       ├── SwissGermanBERT.ipynb
│       ├── Wav2Vec.ipynb
│       ├── XLM-RoBERTa.ipynb
│       ├── byt5.ipynb
│       └── fastText.ipynb
```

## 🔧 Usage

You can either run the Jupyter notebooks for guided experiments, or import the small utilities directly in Python.

### Option A: Run the notebooks
1) Start Jupyter
```bash
jupyter lab    # or: jupyter notebook
```

2) Open the notebooks under `src/notebooks/`:
- `SwissGermanBERT.ipynb` and `GeneralGermanBERT.ipynb` for text classification with Hugging Face models
- `XLM-RoBERTa.ipynb` and `byt5.ipynb` for multilingual experiments
- `Wav2Vec.ipynb` and `AST.ipynb` for audio/speech experiments
- `fastText.ipynb` for classic fastText baselines

The notebooks assume the packages listed in Installation are available. If a cell complains about a missing package, install it into your virtual environment with `pip install <package>` and re-run.

### Option B: Use the utilities in Python

Load and resample audio (mono, 16 kHz) with `data_loader.py`:
```python
from src.utils.data_loader import load_audio_file

waveform, sr = load_audio_file("path/to/audio.wav", target_sample_rate=16000)
print(waveform.shape, sr)
```

Create splits and tabular datasets for text/audio with `prepare_dataset.py`:
```python
from src.utils.prepare_dataset import make_datasets, make_audio_splits

# Example inputs
json_path = "path/to/annotations.json"  # contains entries with dialect texts and ids
dialects = ["ch_lu", "ch_be", "ch_zh"]

# Create Hugging Face Dataset objects for text
train_ds, val_ds, test_ds, dialect2label, train_ids, val_ids, test_ids = make_datasets(json_path, dialects)

# Create Pandas DataFrames listing audio files for each split
audio_root = "/path/to/audio/root"  # expected to have subfolders per dialect (e.g., 'lu', 'be', 'zh')
train_df, val_df, test_df = make_audio_splits(audio_root, dialects, dialect2label, train_ids, val_ids, test_ids)
```

Tip: In your own scripts or notebooks, you can then feed these datasets/tables into Hugging Face `transformers` pipelines or PyTorch `DataLoader`s.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and model hub
- **PyTorch Team**: For the deep learning framework
- **Course Instructors**: For guidance and support

## 📚 Resources

- [Hugging Face Models](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SwissBERT Paper](https://aclanthology.org/2023.swisstext-1.3/)
