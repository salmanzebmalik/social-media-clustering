import numpy as np
from sklearn.cluster import KMeans, DBSCAN

def kmeans_cluster(X: np.ndarray, n_clusters=8, random_state=42):
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km

def dbscan_cluster(X: np.ndarray, eps=0.7, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, db
