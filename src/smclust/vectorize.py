from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(texts, max_features=5000, ngram_range=(1, 2)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(texts)
    return X, vec
