from smclust.vectorize import tfidf_features

def test_tfidf_features_shape():
    X, vec = tfidf_features(["dog cat", "cat mouse"], max_features=10)
    assert X.shape[0] == 2 and X.shape[1] <= 10
