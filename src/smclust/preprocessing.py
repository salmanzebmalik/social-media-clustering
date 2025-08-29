import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return [w for w in text.split() if w not in _stopwords]

def lemmatize(tokens):
    return [_lemmatizer.lemmatize(t) for t in tokens]

def preprocess_df(df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df[text_col].astype(str).map(clean_text)
    df["tokens"] = df["text_clean"].map(tokenize)
    df["lemmas"] = df["tokens"].map(lemmatize)
    return df
