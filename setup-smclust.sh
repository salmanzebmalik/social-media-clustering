#!/usr/bin/env bash
set -euo pipefail

echo "▶ Creating lightweight, reproducible notebooks with Jupytext pairing…"

# 1) Ensure folders
mkdir -p notebooks

# 2) Ensure jupytext is available (local install if needed)
if ! command -v jupytext >/dev/null 2>&1; then
  echo "• Installing jupytext locally (pip)…"
  python -m pip install --quiet jupytext
fi

# 3) Write paired .py notebooks (percent format). Then convert to .ipynb

# 01_data_preprocessing
cat > notebooks/01_data_preprocessing.py <<'PY'
# %% [markdown]
# # 01 – Data Preprocessing
# Loads the raw social media dataset, applies cleaning, tokenization, lemmatization,
# and saves/returns a processed DataFrame. Uses `smclust.preprocessing`.

# %%
from smclust.data_loader import load_messages
from smclust.preprocessing import preprocess_df

df = load_messages()                 # expects data/raw/dataset.json
df = preprocess_df(df)               # adds: text_clean, tokens, lemmas
df.head()
PY

# 02_tfidf_clustering
cat > notebooks/02_tfidf_clustering.py <<'PY'
# %% [markdown]
# # 02 – TF-IDF + Clustering
# Vectorizes cleaned text with TF-IDF, reduces with PCA, clusters (KMeans/DBSCAN),
# and prints evaluation metrics.

# %%
import numpy as np
from smclust.data_loader import load_messages
from smclust.preprocessing import preprocess_df
from smclust.vectorize import tfidf_features
from smclust.reduce import pca_reduce
from smclust.cluster import kmeans_cluster, dbscan_cluster
from smclust.metrics import silhouette, davies_bouldin, dunn_index

# %%
df = load_messages()
df = preprocess_df(df)
X_sparse, vec = tfidf_features(df["text_clean"], max_features=5000, ngram_range=(1,2))
X = X_sparse.toarray()

Xp, pca = pca_reduce(X, n_components=50, random_state=42)

labels_km, _ = kmeans_cluster(Xp, n_clusters=8, random_state=42)
labels_db, _ = dbscan_cluster(Xp, eps=0.7, min_samples=5)

print("KMeans  -> sil:", silhouette(Xp, labels_km),
      " db:", davies_bouldin(Xp, labels_km),
      " dunn:", dunn_index(Xp, labels_km))

print("DBSCAN  -> sil:", silhouette(Xp, labels_db),
      " db:", davies_bouldin(Xp, labels_db),
      " dunn:", dunn_index(Xp, labels_db))
PY

# 03_word2vec_clustering
cat > notebooks/03_word2vec_clustering.py <<'PY'
# %% [markdown]
# # 03 – Word2Vec + Clustering
# Trains Word2Vec on lemmas, builds document vectors, reduces with PCA, clusters,
# and evaluates (Silhouette, Davies–Bouldin, Dunn).

# %%
import numpy as np
from smclust.data_loader import load_messages
from smclust.preprocessing import preprocess_df
from smclust.embeddings import train_word2vec, doc_vector
from smclust.reduce import pca_reduce
from smclust.cluster import kmeans_cluster, dbscan_cluster
from smclust.metrics import silhouette, davies_bouldin, dunn_index

# %%
df = load_messages()
df = preprocess_df(df)

tokenized = df["lemmas"].tolist()
w2v = train_word2vec(tokenized, vector_size=100, window=5, min_count=2, workers=4, seed=42)
X = np.vstack([doc_vector(toks, w2v) for toks in tokenized])

Xp, pca = pca_reduce(X, n_components=50, random_state=42)

labels_km, _ = kmeans_cluster(Xp, n_clusters=8, random_state=42)
labels_db, _ = dbscan_cluster(Xp, eps=0.7, min_samples=5)

print("KMeans  -> sil:", silhouette(Xp, labels_km),
      " db:", davies_bouldin(Xp, labels_km),
      " dunn:", dunn_index(Xp, labels_km))

print("DBSCAN  -> sil:", silhouette(Xp, labels_db),
      " db:", davies_bouldin(Xp, labels_db),
      " dunn:", dunn_index(Xp, labels_db))
PY

# 04_dim_reduction_explainer
cat > notebooks/04_dim_reduction_explainer.py <<'PY'
# %% [markdown]
# # 04 – Dimensionality Reduction Explainer (PCA / t-SNE / UMAP)
# Demonstrates multiple projections of the same feature matrix and basic visual comparison.

# %%
import numpy as np
from smclust.data_loader import load_messages
from smclust.preprocessing import preprocess_df
from smclust.vectorize import tfidf_features
from smclust.reduce import pca_reduce, tsne_embed, umap_embed
from smclust.cluster import kmeans_cluster
from smclust.viz import scatter_2d

# %%
# Build a TF-IDF baseline for visual projections
df = load_messages()
df = preprocess_df(df)
X_sparse, _ = tfidf_features(df["text_clean"], max_features=3000)
X = X_sparse.toarray()

# %%
Xp, pca = pca_reduce(X, n_components=50, random_state=42)
labels, _ = kmeans_cluster(Xp, n_clusters=8, random_state=42)

# PCA 2D (using the first two PCs of the reduced space)
pca2d = Xp[:, :2]
scatter_2d(pca2d, labels, title="PCA (first two components)")

# t-SNE
tsne2d = tsne_embed(Xp, n_components=2, perplexity=30, random_state=42)
scatter_2d(tsne2d, labels, title="t-SNE (perplexity=30)")

# UMAP
umap2d = umap_embed(Xp, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
scatter_2d(umap2d, labels, title="UMAP (n_neighbors=15, min_dist=0.1)")
PY

# 4) Pair & convert to real notebooks
echo "• Converting .py notebooks to .ipynb with jupytext…"
jupytext --to ipynb notebooks/01_data_preprocessing.py
jupytext --to ipynb notebooks/02_tfidf_clustering.py
jupytext --to ipynb notebooks/03_word2vec_clustering.py
jupytext --to ipynb notebooks/04_dim_reduction_explainer.py

echo "✅ Notebooks created in notebooks/:"
ls -1 notebooks/*.ipynb
