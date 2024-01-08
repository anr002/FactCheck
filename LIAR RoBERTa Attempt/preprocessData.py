import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Clean and Preprocess strings. Remove punctuation convert to lowercase
def clean_txt(text):
    text = re.sub("'", "", text)
    text = re.sub("(\\W)+", " ", text)
    return text.lower()

# Splitting strings, no longer than 512 tokens since this is BERT's maxed seq length
def get_split(text, max_seq_len):
    l_total = []
    l_partial = []
    if len(text.split()) // max_seq_len > 0:
        n = len(text.split()) // max_seq_len
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_partial = text.split()[:max_seq_len]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = text.split()[w * max_seq_len: w * max_seq_len + max_seq_len]
            l_total.append(" ".join(l_partial))
    return l_total

# Organizing DataFrame and appending data
def data_augmentation(df):
    text_l, label_l, index_l = [], [], []
    for idx, row in df.iterrows():
        for l in row['text_split']:
            text_l.append(l)
            label_l.append(row['label'])
            index_l.append(idx)
    return pd.DataFrame({'text': text_l, 'label': label_l, 'index': index_l})

# Load and preprocess datasets
def loadPreprocessData(filepath, max_seq_len):
    train_df = pd.read_csv(filepath)
    train_df['text'] = train_df['text'].apply(lambda x: clean_txt(str(x)))
    train_df['text_split'] = train_df['text'].apply(lambda x: get_split(x, max_seq_len))

    # Create train and validate datasets
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Applying data augmentation. Reset index cause of KeyError: 19561
    train_df_augmented = data_augmentation(train_df).reset_index(drop=True)
    val_df_augmented = data_augmentation(val_df).reset_index(drop=True)

    return train_df_augmented, val_df_augmented