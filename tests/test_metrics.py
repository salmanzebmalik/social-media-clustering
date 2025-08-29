import numpy as np
from smclust.metrics import silhouette, davies_bouldin, dunn_index

def test_metrics_return_numbers():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3)
    labels = np.array([0]*10 + [1]*10)
    assert not np.isnan(silhouette(X, labels))
    assert not np.isnan(davies_bouldin(X, labels))
    # dunn can be NaN if degenerate; create separated clusters
    X2 = np.vstack([rng.randn(10,3)-3, rng.randn(10,3)+3])
    labels2 = np.array([0]*10 + [1]*10)
    assert not np.isnan(dunn_index(X2, labels2))
