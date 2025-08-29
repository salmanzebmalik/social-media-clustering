from smclust.embeddings import train_word2vec, doc_vector
import numpy as np

def test_w2v_and_doc_vector():
    toks = [["hello","world"],["hello","dog"]]
    model = train_word2vec(toks, vector_size=20, min_count=1, workers=1, window=2, seed=42)
    v = doc_vector(["hello","world"], model)
    assert isinstance(v, np.ndarray) and v.shape == (20,)
