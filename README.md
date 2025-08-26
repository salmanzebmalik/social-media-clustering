# 📊 Social Media Clustering – Unsupervised Learning Case Study

## 📌 Project Overview  
This project analyzes a dataset of **social media posts and user relationships** using **unsupervised learning techniques**.  
The goal was to uncover hidden structures in the data by clustering posts and users, validating the results with multiple evaluation metrics, and visualizing patterns.  

The work was carried out as part of the **Unsupervised Learning / Data Analytics 1 (WT 2024/25)** module at the University of Münster.

---

## 🎯 Objectives  
- Preprocess and embed social media text data.  
- Apply **Word2Vec** for semantic text representation.  
- Reduce dimensions using **PCA** and **t-SNE** for visualization.  
- Cluster embeddings using **K-means** and **DBSCAN**.  
- Evaluate clusters with **Silhouette, Dunn, and Davies-Bouldin indices**.  
- Explore the **user network (graph structure)** based on friendship/following relations.  
- Present results visually (poster + figures).  

---

## 📂 Project Structure  
```

project-name/
│
├── README.md                <- Project overview (this file)
├── requirements.txt         <- Python dependencies
├── environment.yml          <- Conda environment spec (optional)
├── .gitignore               <- Ignore datasets, models, logs
│
├── data/
│   ├── raw/                 <- Original dataset.json & user\_relations.csv
│   ├── processed/           <- Cleaned datasets
│   └── external/            <- Benchmark datasets (if any)
│
├── notebooks/
│   ├── 01\_data\_preprocessing.ipynb
│   ├── 02\_exploratory\_analysis.ipynb
│   ├── 03\_model\_training.ipynb
│   └── 04\_evaluation.ipynb
│
├── src/
│   ├── **init**.py
│   ├── data\_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/                  <- Trained models (excluded from GitHub via .gitignore)
│
├── results/
│   ├── figures/             <- Cluster diagrams, embeddings plots, network graphs
│   └── metrics.json         <- Scores from evaluation metrics
│
├── app/                     <- (optional) Streamlit/Docker deployment
│   ├── streamlit\_app.py
│   ├── dockerfile
│   └── requirements.txt
│
└── tests/                   <- Unit tests
├── test\_preprocessing.py
├── test\_train.py
└── test\_evaluate.py

````

---

## 🔧 Installation & Setup  
Clone the repository:  
```bash
git clone https://github.com/<your-username>/social-media-clustering.git
cd social-media-clustering
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate sm-clustering
```

---

## 🚀 Usage

Run notebooks step by step:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

Run scripts directly:

```bash
python src/train.py
python src/evaluate.py
```

For visualization app (optional):

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Results

* Word2Vec embeddings revealed meaningful groupings of posts.
* PCA and t-SNE reduced dimensions for interpretable visualizations.
* **K-means** captured compact clusters; **DBSCAN** detected anomalies/noise.
* Cluster validation showed **Silhouette \~0.42** (moderate separation).
* Network analysis revealed distinct user groups with snowflake-like community structures.

Example visualization:
![Cluster Visualization](results/figures/clusters_example.png)

---

## 🛠️ Tech Stack

* **Languages**: Python (Pandas, NumPy, scikit-learn, Gensim)
* **ML/Clustering**: Word2Vec, PCA, t-SNE, K-means, DBSCAN
* **Visualization**: Matplotlib, Seaborn, NetworkX
* **Deployment**: Streamlit (optional)

---

## 📌 Acknowledgements

* Project assignment by **Prof. Christian Grimme & Janina Lütke Stockdiek** (University of Münster).
* Dataset: Provided in the course (social media messages + user relations).

---

## 🌟 Professional Highlight

This project demonstrates hands-on skills in:

* **Text embeddings & dimensionality reduction**
* **Clustering algorithms & evaluation metrics**
* **Network analysis and visualization**
* **Reproducible ML pipelines and clean project structure**

It shows both **data science workflow mastery** and **software engineering best practices** (unit testing, modular code, deployment option).

---

