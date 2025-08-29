import argparse
import json
import numpy as np
from smclust.config import RESULTS, FIGURES, RANDOM_STATE, N_CLUSTERS, EPS, MIN_SAMPLES
from smclust.data_loader import load_messages
from smclust.preprocessing import preprocess_df
from smclust.vectorize import tfidf_features
from smclust.embeddings import train_word2vec, doc_vector
from smclust.reduce import pca_reduce
from smclust.cluster import kmeans_cluster, dbscan_cluster
from smclust.metrics import silhouette, davies_bouldin, dunn_index

def main(args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_messages()
    df = preprocess_df(df)

    if args.route == "tfidf":
        X, vec = tfidf_features(df["text_clean"])
        X = X.toarray()
    else:
        tokenized = df["lemmas"].tolist()
        w2v = train_word2vec(tokenized, vector_size=args.vector_size, seed=RANDOM_STATE)
        X = np.vstack([doc_vector(toks, w2v) for toks in tokenized])

    Xp, _ = pca_reduce(X, n_components=args.pca_components)

    if args.algo == "kmeans":
        labels, _ = kmeans_cluster(Xp, n_clusters=args.n_clusters, random_state=RANDOM_STATE)
    else:
        labels, _ = dbscan_cluster(Xp, eps=args.eps, min_samples=args.min_samples)

    metrics = {
        "algo": args.algo,
        "route": args.route,
        "silhouette": float(silhouette(Xp, labels)),
        "davies_bouldin": float(davies_bouldin(Xp, labels)),
        "dunn": float(dunn_index(Xp, labels)),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved:", RESULTS / "metrics.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--route", choices=["tfidf", "word2vec"], default="tfidf")
    p.add_argument("--algo", choices=["kmeans", "dbscan"], default="kmeans")
    p.add_argument("--n_clusters", type=int, default=N_CLUSTERS)
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--min_samples", type=int, default=MIN_SAMPLES)
    p.add_argument("--vector_size", type=int, default=100)
    p.add_argument("--pca_components", type=int, default=50)
    args = p.parse_args()
    main(args)
