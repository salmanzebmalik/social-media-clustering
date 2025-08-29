from smclust.preprocessing import clean_text, preprocess_df
import pandas as pd

def test_clean_text_basic():
    s = "Hello!!! Visit https://a.b/c NOW"
    out = clean_text(s)
    assert "http" not in out and out.islower()

def test_preprocess_df_columns():
    df = pd.DataFrame({"text": ["Cats & dogs", "Links: http://x.y"]})
    out = preprocess_df(df)
    assert {"text_clean","tokens","lemmas"}.issubset(out.columns)
