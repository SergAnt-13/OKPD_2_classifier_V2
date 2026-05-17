from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Данные
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATA_DIR = DATA_DIR / "training"
REFERENCE_DIR = DATA_DIR / "reference"          # вместо TAXONOMY_DIR

# Артефакты (только актуальные компоненты)
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CLASSIFIER_DIR = ARTIFACTS_DIR / "classifier"
BIENCODER_DIR = ARTIFACTS_DIR / "bi_encoder"
FAISS_DIR = ARTIFACTS_DIR / "faiss"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# История экспериментов
RUNS_DIR = ROOT_DIR / "runs"

# Создаём только то, что реально нужно
REQUIRED_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAINING_DATA_DIR,
    REFERENCE_DIR,
    CLASSIFIER_DIR,
    BIENCODER_DIR,
    FAISS_DIR,
    METRICS_DIR,
    RUNS_DIR,
]

for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)