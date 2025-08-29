PY=python

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

format:
	$(PY) -m black src tests
	$(PY) -m isort src tests

lint:
	$(PY) -m ruff check src tests

test:
	$(PY) -m pytest

run-kmeans:
	$(PY) scripts/run_pipeline.py --route w2v --algo kmeans --n_clusters 8

run-dbscan:
	$(PY) scripts/run_pipeline.py --route w2v --algo dbscan --eps 0.6 --min_samples 5
