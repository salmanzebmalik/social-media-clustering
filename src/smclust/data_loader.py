import json
import pandas as pd
from .config import RAW

def load_messages(path=RAW / "dataset.json") -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.json_normalize(data)

def load_user_relations(path=RAW / "user_relations.csv") -> pd.DataFrame:
    return pd.read_csv(path, names=["source", "target"])
