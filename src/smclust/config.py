from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
EXTERNAL = DATA_DIR / "external"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = PROJECT_ROOT / "models"

# Default params
N_CLUSTERS = 8
EPS = 0.7
MIN_SAMPLES = 5
RANDOM_STATE = 42
