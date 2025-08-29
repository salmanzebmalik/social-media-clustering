import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def pca_reduce(X: np.ndarray, n_components=50, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X), pca

def tsne_embed(X: np.ndarray, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(X)

def umap_embed(X: np.ndarray, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)
