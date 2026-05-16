from pathlib import Path

# ROOT
ROOT_DIR = Path(__file__).resolve().parent.parent

# DATA
DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATA_DIR = DATA_DIR / "training"
INFERENCE_DATA_DIR = DATA_DIR / "inference"
TAXONOMY_DIR = DATA_DIR / "taxonomy"
PSEUDO_LABELS_DIR = DATA_DIR / "pseudo_labels"

# ARTIFACTS
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

BIENCODER_DIR = ARTIFACTS_DIR / "biencoder"
CLASSIFIER_DIR = ARTIFACTS_DIR / "classifier"
FAISS_DIR = ARTIFACTS_DIR / "faiss"
RUNS_DIR = ARTIFACTS_DIR / "runs"

# LOGS
LOGS_DIR = ROOT_DIR / "logs"

# CREATE DIRS
ALL_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAINING_DATA_DIR,
    INFERENCE_DATA_DIR,
    TAXONOMY_DIR,
    PSEUDO_LABELS_DIR,
    BIENCODER_DIR,
    CLASSIFIER_DIR,
    FAISS_DIR,
    RUNS_DIR,
    LOGS_DIR
]

for directory in ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)