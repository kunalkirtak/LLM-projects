"""
Benchmark dataset loading.

The dataset is a plain CSV so it is easy to inspect, version-control, and
extend by hand. Reference keywords are stored pipe-separated inside a single
CSV cell (e.g. "mitochondria|ATP|organelle") and expanded into a list here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from models.evaluation import BenchmarkItem

DEFAULT_DATASET_PATH = Path("data/benchmark_dataset.csv")


def load_benchmark(path: str | Path = DEFAULT_DATASET_PATH) -> List[BenchmarkItem]:
    """Load and validate the benchmark dataset from a CSV file.

    Args:
        path: Path to the benchmark CSV. Defaults to the bundled sample set.

    Returns:
        A list of validated BenchmarkItem objects.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at '{path}'")

    df = pd.read_csv(path)

    required_columns = {"question", "expected_answer", "category", "difficulty", "reference_keywords"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark dataset is missing required columns: {sorted(missing)}")

    items: List[BenchmarkItem] = []
    for idx, row in df.iterrows():
        keywords_raw = str(row.get("reference_keywords", "") or "")
        keywords = [kw.strip() for kw in keywords_raw.split("|") if kw.strip()]

        try:
            item = BenchmarkItem(
                id=int(idx),
                question=str(row["question"]).strip(),
                expected_answer=str(row["expected_answer"]).strip(),
                category=str(row["category"]).strip(),
                difficulty=str(row["difficulty"]).strip(),
                reference_keywords=keywords,
            )
            items.append(item)
        except Exception as exc:  # noqa: BLE001 - we want to log and skip bad rows
            logger.warning(f"Skipping malformed benchmark row {idx}: {exc}")

    logger.info(f"Loaded {len(items)} benchmark items from {path}")
    return items


def dataset_to_dataframe(items: List[BenchmarkItem]) -> pd.DataFrame:
    """Convert benchmark items back into a DataFrame, useful for display in Streamlit."""
    return pd.DataFrame(
        [
            {
                "id": item.id,
                "question": item.question,
                "expected_answer": item.expected_answer,
                "category": item.category,
                "difficulty": item.difficulty,
                "reference_keywords": ", ".join(item.reference_keywords),
            }
            for item in items
        ]
    )
