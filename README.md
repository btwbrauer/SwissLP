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
- **Whisper**: OpenAI's robust speech recognition model

### Hardware Support
- **NVIDIA GPUs**: Automatic CUDA support
- **AMD GPUs**: Automatic ROCm support
- **Apple Silicon**: MPS acceleration on macOS
- **CPU**: Fallback for all systems

## 🛠️ Installation

### Option 1: Nix (Recommended - Reproducible)

The Nix approach provides the most reproducible environment across all platforms.

#### Prerequisites: Install Nix

**Linux (including WSL2):**
```bash
# Install Nix
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- --no-confirm

# Enable flakes (add to ~/.config/nix/nix.conf)
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# Restart your shell or run:
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
```

**macOS:**
```bash
# Install Nix
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- --no-confirm

# Enable flakes (add to ~/.config/nix/nix.conf)
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# Restart your shell or run:
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
```

**Windows (WSL2):**
```bash
# In WSL2, follow the Linux instructions above
# Make sure WSL2 is updated: wsl --update
```

#### Quick Start with Nix

1. **Clone the repository:**
   ```bash
   git clone https://github.com/btwbrauer/SwissLP.git
   cd SwissLP
   ```

2. **Enter the development environment:**
   ```bash
   nix develop
   ```

   The flake automatically detects your hardware and installs the appropriate PyTorch version.

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   pytest tests/ -v
   ```

#### Hardware-Specific Environments

```bash
# For NVIDIA CUDA
nix develop .#cuda

# For AMD ROCm
nix develop .#rocm

# For CPU-only (default)
nix develop .#default
```

### Option 2: Docker (Alternative)

If you prefer Docker or cannot install Nix:

```bash
# Build the Docker image
nix build
docker load < result

# Run the container
docker run -it --gpus all swisslp:latest
```

### Option 3: Python Virtual Environment (Fallback)

**Note:** This method may have version conflicts and is less reproducible.

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   pytest tests/ -v
   ```

## 📁 Project Structure

```
SwissLP/
├── flake.nix              # Nix flake for reproducible environment
├── requirements.txt       # Python dependencies (exact versions)
├── pytest.ini            # Pytest configuration
├── README.md             # This file
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_models.py    # Text model loaders
│   │   └── speech_models.py  # Speech model loaders
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py    # Data loading utilities
├── tests/
│   ├── __init__.py
│   ├── test_text_models.py
│   ├── test_speech_models.py
│   └── test_data_loader.py
└── notebooks/            # Jupyter notebooks for experiments
```

## 🔧 Usage

### Text Models

#### SwissBERT (Swiss German BERT)
```python
from src.models import load_swissbert

# Load pre-trained SwissBERT
model, tokenizer = load_swissbert()

# Load for classification with custom number of labels
model, tokenizer = load_swissbert(num_labels=4)

# Text preprocessing and inference
text = "Grüezi, wie gaht's dir hüt?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

#### German BERT
```python
from src.models import load_german_bert

# Load German BERT
model, tokenizer = load_german_bert()

# Load for classification
model, tokenizer = load_german_bert(num_labels=3)

# Inference
text = "Hallo, wie geht es dir heute?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

#### XLM-RoBERTa (Multilingual)
```python
from src.models import load_xlm_roberta

# Load XLM-RoBERTa
model, tokenizer = load_xlm_roberta()

# Load for classification
model, tokenizer = load_xlm_roberta(num_labels=5)

# Multilingual inference
texts = [
    "Grüezi, wie gaht's?",  # Swiss German
    "Hallo, wie geht's?",   # German
    "Hello, how are you?"   # English
]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
```

#### ByT5 (Byte-level T5)
```python
from src.models import load_byt5

# Load ByT5
model, tokenizer = load_byt5()

# Load specific variant
model, tokenizer = load_byt5(model_name="google/byt5-small")

# Text generation/translation
text = "Grüezi, wie gaht's?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Speech Models

#### Wav2Vec2 (Speech Classification)
```python
from src.models import load_wav2vec2
import torchaudio

# Load Wav2Vec2
model, feature_extractor = load_wav2vec2()

# Load for classification
model, feature_extractor = load_wav2vec2(num_labels=4)

# Audio preprocessing and inference
audio_path = "audio.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Resample to 16kHz if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Extract features
inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

#### AST (Audio Spectrogram Transformer)
```python
from src.models import load_ast
import librosa

# Load AST
model, feature_extractor = load_ast()

# Load for classification
model, feature_extractor = load_ast(num_labels=3)

# Audio preprocessing
audio_path = "audio.wav"
waveform, sample_rate = librosa.load(audio_path, sr=16000)

# Extract features
inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

#### Whisper (Speech Recognition)
```python
from src.models import load_whisper
import torchaudio

# Load Whisper
model, processor = load_whisper()

# Load specific variant
model, processor = load_whisper(model_name="openai/whisper-base")

# Audio preprocessing and transcription
audio_path = "audio.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Process audio
inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs)
    transcription = processor.decode(outputs[0], skip_special_tokens=True)
```

### Data Loading and Preprocessing

#### Audio Data
```python
from src.utils import load_audio_file, preprocess_audio

# Load audio file
waveform, sample_rate = load_audio_file("audio.wav")

# Preprocess audio (resample, normalize)
processed_audio = preprocess_audio(waveform, sample_rate, target_sr=16000)

# Create audio dataset
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
labels = [0, 1, 0]
dataloader = create_audio_dataloader(audio_files, labels, batch_size=8)
```

#### Text Data
```python
from src.utils import create_text_dataloader, preprocess_text, split_dataset

# Text preprocessing
text = "Grüezi, wie gaht's dir hüt?"
cleaned_text = preprocess_text(text, lowercase=True, remove_punctuation=True)

# Create text dataset
texts = ["Text 1", "Text 2", "Text 3"]
labels = [0, 1, 0]
dataloader = create_text_dataloader(texts, labels, batch_size=16)

# Split dataset
train_data, val_data, test_data = split_dataset(
    data=texts,
    labels=labels,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### Load All Models

```python
from src.models import load_all_text_models, load_all_speech_models

# Load all text models
text_models = load_all_text_models()
# Returns: {'swissbert': {'model': ..., 'tokenizer': ...}, 'german_bert': ..., ...}

# Load all speech models
speech_models = load_all_speech_models()
# Returns: {'wav2vec2': {'model': ..., 'processor': ...}, 'ast': ..., ...}

# Use specific models
swissbert_model = text_models['swissbert']['model']
wav2vec2_model = speech_models['wav2vec2']['model']
```

### Device Management

```python
from src.models import get_device
import torch

# Get optimal device
device = get_device()
print(f"Using device: {device}")

# Move models to device
model = model.to(device)

# Move data to device
inputs = {k: v.to(device) for k, v in inputs.items()}
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_text_models.py -v
pytest tests/test_speech_models.py -v
pytest tests/test_data_loader.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 🔬 Development

### Code Quality

The project includes:
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

Format code:
```bash
black src/ tests/
```

Lint code:
```bash
flake8 src/ tests/
```

Type check:
```bash
mypy src/
```

### Jupyter Notebooks

Start Jupyter:
```bash
jupyter notebook
```

Or use Jupyter Lab:
```bash
jupyter lab
```

## 📦 Dependencies

### Exact Versions (from flake.nix)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.7.0 | Deep learning framework |
| torchaudio | 2.7.0a0 | Audio processing |
| torchvision | 0.22.0 | Computer vision |
| transformers | 4.51.3 | Hugging Face models |
| datasets | 3.5.1 | Dataset utilities |
| tokenizers | 0.21.1 | Text tokenization |
| accelerate | 1.5.2 | Training acceleration |
| nltk | 3.9.1 | Natural language toolkit |
| sentencepiece | 0.2.0 | Text tokenization |
| librosa | 0.11.0 | Audio analysis |
| soundfile | 0.13.1 | Audio I/O |
| numpy | 2.2.5 | Numerical computing |
| pandas | 2.2.3 | Data manipulation |
| scikit-learn | 1.6.1 | Machine learning |
| jupyter | 1.1.1 | Notebook environment |
| pytest | 8.3.5 | Testing framework |
| pytest-cov | 6.1.0 | Coverage reporting |
| tqdm | 4.67.1 | Progress bars |
| pyyaml | 6.0.2 | YAML parsing |

See `requirements.txt` for the complete list with exact versions.

## 🌍 Hardware Optimization

The Nix flake automatically selects the optimal PyTorch build:

| Platform | Hardware | PyTorch Build | Command |
|----------|----------|---------------|---------|
| Linux | NVIDIA GPU | torchWithCuda | `nix develop .#cuda` |
| Linux | AMD GPU | torchWithRocm | `nix develop .#rocm` |
| macOS | Apple Silicon | torch (with MPS) | `nix develop` |
| macOS | Intel | torch (CPU) | `nix develop` |
| Other | Any | torch (CPU) | `nix develop` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Format code: `black src/ tests/`
6. Submit a pull request

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and model hub
- **NixOS Community**: For the reproducible build system
- **PyTorch Team**: For the deep learning framework
- **Course Instructors**: For guidance and support

## 📚 Resources

- [Hugging Face Models](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Nix Flakes](https://nixos.wiki/wiki/Flakes)
- [SwissBERT Paper](https://aclanthology.org/2023.swisstext-1.3/)