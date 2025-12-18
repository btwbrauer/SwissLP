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
- **fastText**: lightweight model using word and character n-grams as a non-transformer baseline

### Speech Models
- **Wav2Vec2**: Facebook's self-supervised speech representation
- **AST**: Audio Spectrogram Transformer


## How to Run

This repository is focused on running notebooks for comparing how the different AI models handle different Swiss German dialects. The simplest way to get started is by using [Google Colab](https://colab.google/).


### 1. Open a Notebook in Colab
In Colab: Open Notebook -> GitHub -> `https://github.com/btwbrauer/SwissLP`

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

### 4. Running the Notebooks
With the steps above, you are ready to run the Jupyter notebooks.
Each model has its own notebook, which can be run in any order and includes some comments.
All notebooks were tested on a Google Colab GPU runtime, and example results are already visible in the outputs, so running the code is not necessarily required.

## Authors
- Justin Verhoek
- Björn Brauer
- Thomas Joos
- Marion Andermatt

## Resources

- [Hugging Face Models](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SwissBERT Paper](https://aclanthology.org/2023.swisstext-1.3/)
