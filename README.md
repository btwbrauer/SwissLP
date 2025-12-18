# Swiss Language Processing (SwissLP)
This repository supplements the project "Comparative Study of Pre-Trained Models for Swiss German Dialect Classification", which evaluates several text-based and speech-based pre-trained models on the task of Swiss German dialect Classification.

Swiss German is a continuum of closely related dialects, making dialect classification a challenging low-resource and fine-grained task. The goal of this project is to determine which type of pre-training, domain-specific, general-purpose, or speech-based, is most effective for this task.

## Table of Contents
- [Models](#models)
- [Dataset](#dataset)
- [Dataset Preprocessing](#dataset-preprocessing)
- [Model Training](#model-training)
- [How to Run](#how-to-run)
- [Results](#results)
- [Discussion](#discussion)
- [Authors](#authors)
- [Resources](#resources)


## Models
### Text Models
- **SwissBERT**: BERT model fine-tuned on Swiss German data
- **General German BERT**: Standard BERT model for German language
- **XLM-RoBERTa**: Cross-lingual model supporting 100+ languages
- **ByT5**: Byte-level T5 model for multilingual tasks
- **fastText**: Lightweight model using word and character n-grams as a non-transformer baseline

### Speech Models
- **Wav2Vec2**: Facebook's self-supervised speech representation
- **AST**: Audio Spectrogram Transformer

## Dataset
For this project, the SwissDial Dataset povided by the Media Technology Center at ETH Zurich was used. It can be downloaded here: (https://mtc.ethz.ch/publications/open-source/swiss-dial.html)

The dataset contains about 2700 voice recordings across eight major Swiss German dialects: Aargau, Bern, Basel, Lucerne, St. Gallen, Graubünden, Valais and Zurich. Each recording is between 3 and 12 seconds long, and there is one speaker per dialect. 

The dataset provides a JSON file with transcripts of all recordings.

## Dataset Preprocessing
To ensure consistency across dialects, only recordings and transcripts for which corresponding data existed for all eight dialects were included in the experiments. Furthermore, in the transcripts, quotation marks were standardized to a single double-quote character. No other normalization was performed.

The data was split the following way: 80% Train, 10% Validation and 10% Test.

## Model Training
All models were trained using the same balanced train, validation, and test splits to ensure comparability across the different models. Transformer-based models were fine-tuned from pre-trained checkpoints using standard optimization settings, while fastText was trained from scratch on the training data

## How to Run

This repository is focused on running notebooks for comparing how the different LLMs handle different Swiss German dialects. The simplest way to get started is by using [Google Colab](https://colab.google/).


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

## Results
The macro-average precision, recall and F1-score were measured for each experiment.

| Model Type   | Model                     | Avg. Precision | Avg. Recall | Avg. F1-Score |
|-------------|---------------------------|---------------|-------------|---------------|
| Text-based  | General German BERT        | 0.89          | 0.89        | 0.89          |
| Text-based  | SwissBERT                 | 0.88          | 0.87        | 0.87          |
| Text-based  | XLM-R (XLM-RoBERTa)        | 0.85          | 0.84        | 0.84          |
| Text-based  | ByT5                      | 0.77          | 0.76        | 0.76          |
| Text-based  | fastText                  | 0.72          | 0.72        | 0.71          |
| Speech-based| Wav2Vec2                  | 1.00          | 1.00        | 1.00          |
| Speech-based| AST                       | 1.00          | 1.00        | 1.00          |

## Discussion
The General German BERT model performed the best among the text-based models. The SwissBERT model performed only sloghtly worse, indicating that strong general prupose German pre-training is already highly effective for dialect classification.

Both speech-based models achieved perfect scores across all three metrics. This is likely due to the fact that the SwissDial dataset has only one speaker per dialect, resulting in the possibility that the speech-based model exploit speaker-specific characteristics. 

This shows the strength of general-purpose text pre-training for Swiss German dialect classification and the necessity for a multi-speaker dataset when evaluation speech-based models.

## Authors
- Justin Verhoek
- Björn Brauer
- Thomas Joos
- Marion Andermatt

## Resources
- [Hugging Face Models](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SwissBERT Paper](https://aclanthology.org/2023.swisstext-1.3/)
- [SwissDial Dataset](https://mtc.ethz.ch/publications/open-source/swiss-dial.html)
- [SwissDial: Parallel Multidialect Corpus of Spoken Swiss German](https://arxiv.org/abs/2103.11401)
