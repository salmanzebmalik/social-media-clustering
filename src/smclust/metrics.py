import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def silhouette(X, labels):
    if len(set(labels)) <= 1:
        return np.nan
    return silhouette_score(X, labels)

def davies_bouldin(X, labels):
    if len(set(labels)) <= 1:
        return np.nan
    return davies_bouldin_score(X, labels)

def dunn_index(X, labels):
    try:
        from scipy.spatial.distance import pdist, squareform
        X = np.asarray(X)
        unique = [c for c in set(labels) if c != -1]
        if len(unique) < 2:
            return np.nan
        clusters = [X[labels == c] for c in unique]
        intra = max([pdist(c).max() if len(c) > 1 else 0. for c in clusters])
        centers = [c.mean(axis=0) for c in clusters]
        inter = squareform(pdist(np.vstack(centers))).astype(float)
        inter[inter == 0.] = np.inf
        return inter.min() / (intra if intra > 0 else np.nan)
    except Exception:
        return np.nan
