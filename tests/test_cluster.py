import numpy as np
from smclust.cluster import kmeans_cluster, dbscan_cluster

def test_kmeans_cluster_labels():
    X = np.random.RandomState(0).randn(30, 4)
    labels, model = kmeans_cluster(X, n_clusters=3, random_state=42)
    assert len(labels) == 30 and len(set(labels)) == 3

def test_dbscan_cluster_labels():
    X = np.random.RandomState(0).randn(30, 4)
    labels, model = dbscan_cluster(X, eps=0.8, min_samples=3)
    assert len(labels) == 30
