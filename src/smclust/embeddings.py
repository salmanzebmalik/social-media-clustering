from gensim.models import Word2Vec
import numpy as np

def train_word2vec(tokenized_docs, vector_size=100, window=5, min_count=2, workers=4, seed=42):
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )
    return model

def doc_vector(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0) if len(vecs) else np.zeros(model.vector_size)
