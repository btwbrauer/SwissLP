# 🇨🇭 Swiss Language Processing (SwissLP)

A minimal, best-practice repository for Text and Speech Processing using Swiss German and German language models.

## Course: Essentials in Text and Speech Processing
...

## Features

### Text Models
- **SwissBERT**: BERT model fine-tuned on Swiss German data
- **German BERT**: Standard BERT model for German language
- **XLM-RoBERTa**: Cross-lingual model supporting 100+ languages
- **ByT5**: Byte-level T5 model for multilingual tasks

### Speech Models
- **Wav2Vec2**: Facebook's self-supervised speech representation
- **AST**: Audio Spectrogram Transformer


## How to Run

This repository is focused on running notebooks for comparing how the different AI models handle different Swiss German dialects. The simplest way to get started is by using [Google Colab](https://colab.google/).


### 1. Open a Notebook in Colab
In Colab: Upload Notebook -> GitHub -> `https://github.com/btwbrauer/SwissLP`

### 2. Start a Colab Notebook with a GPU
In Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU


### 3. Upload Data (Google Drive or Custom Path)

#### a. Upload Data to Google Drive

Upload your data to the following location in **Google Drive**:

```
MyDrive/
└── Colab Notebooks/
    └── ETSP/
        ├── data/
        │   ├── ch_sg/
        │   │   ├── *.wav
        │   ├── ch_be/
        │   │   ├── *.wav
        │   ├── ch_gr/
        │   │   ├── *.wav
        │   └── ...
        │
        ├── utils/
        │   ├── __init__.py
        │   ├── data_loader.py
        │   └── prepare_dataset.py
        │
        └── sentences_ch_de_transcribed.json
```

#### b. Custom File Upload
If you store your data in a different location, update the file path in the notebook:
```
src_folder = "YOUR_CUSTOM_PATH"
```
or
```
file_path = "YOUR_CUSTOM_PATH"
```

## Authors
- Justin Verhoek
- Björn Brauer
- Thomas Joos
- Marion Andermatt

## Acknowledgments

- **Hugging Face**: For the transformers library and model hub
- **PyTorch Team**: For the deep learning framework
- **Course Instructors**: For guidance and support

## Resources

- [Hugging Face Models](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SwissBERT Paper](https://aclanthology.org/2023.swisstext-1.3/)
