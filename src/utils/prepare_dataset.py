import json, random, pandas as pd, numpy as np
from datasets import Dataset
import re
import os

def load_data(file_path, dialects):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [ex for ex in data if all(d in ex for d in dialects)]

def split_data(filtered, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(filtered)
    N = len(filtered)
    train = filtered[:int(0.8*N)]
    val = filtered[int(0.8*N):int(0.9*N)]
    test = filtered[int(0.9*N):]
    return train, val, test

def normalize_quotes(text):
    text = re.sub(r'[«»“”]', '"', text)
    text = re.sub(r'"\s*', '"', text)
    text = re.sub(r'\s*"', '"', text)
    return text

def flatten_examples(subset, dialects, dialect2label):
    rows = []
    for ex in subset:
        for d in dialects:
            clean_text = normalize_quotes(ex[d])
            rows.append({'text': clean_text, 'label': dialect2label[d], 'id': ex['id'], 'dialect': d})
    return pd.DataFrame(rows)

def get_split_ids(subset):
    return sorted({ex['id'] for ex in subset})

def make_datasets(file_path, dialects):
    filtered = load_data(file_path, dialects)
    train, val, test = split_data(filtered)
    dialect2label = {d: i for i, d in enumerate(dialects)}
    train_df = flatten_examples(train, dialects, dialect2label)
    val_df = flatten_examples(val, dialects, dialect2label)
    test_df = flatten_examples(test, dialects, dialect2label)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    train_ids, val_ids, test_ids = get_split_ids(train), get_split_ids(val), get_split_ids(test)
    return train_dataset, val_dataset, test_dataset, dialect2label, train_ids, val_ids, test_ids

def make_audio_df(split_ids, dialects, dialect2label, audio_root):
    rows = []
    for d in dialects:
        folder = os.path.join(audio_root, d[3:])  # e.g., 'lu' from 'ch_lu'
        for id in split_ids:
            fname = f"{d}_{id:04}.wav"
            fpath = os.path.join(folder, fname)
            rows.append({"audio_path": fpath, "label": dialect2label[d], "id": id, "dialect": d})
    return pd.DataFrame(rows)

def make_audio_splits(audio_root, dialects, dialect2label, train_ids, val_ids, test_ids):
    train_audio_df = make_audio_df(train_ids, dialects, dialect2label, audio_root)
    val_audio_df = make_audio_df(val_ids, dialects, dialect2label, audio_root)
    test_audio_df = make_audio_df(test_ids, dialects, dialect2label, audio_root)
    return train_audio_df, val_audio_df, test_audio_df
