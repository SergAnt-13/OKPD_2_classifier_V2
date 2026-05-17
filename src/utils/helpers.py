# src/utils/helpers.py
import numpy as np
from pathlib import Path
from typing import List


def softmax(logits: np.ndarray) -> np.ndarray:
    """Стабильный softmax."""
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def ensure_dir(path: Path) -> Path:
    """Создаёт директорию, если её нет."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_texts_from_file(file_path: Path, column: str = "Номенклатура") -> List[str]:
    """Загружает тексты из CSV/Excel файла."""
    import pandas as pd
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df[column].astype(str).tolist()