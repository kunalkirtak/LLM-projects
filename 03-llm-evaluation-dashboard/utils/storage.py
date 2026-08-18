"""
Storage layer: persists evaluation results to SQLite and exports them to
CSV / JSON reports. Keeping this separate from evaluator.py means the
scoring logic has zero knowledge of how/where results end up, which makes
both sides easier to test independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from models.database import EvaluationRecord, get_session, init_db
from models.evaluation import EvaluationResult

RESULTS_CSV_PATH = Path("results/evaluation_results.csv")
REPORT_JSON_PATH = Path("reports/evaluation_report.json")


def save_evaluation(result: EvaluationResult) -> EvaluationResult:
    """Persist a single evaluation result to the SQLite database."""
    init_db()
    session = get_session()
    try:
        record = EvaluationRecord(
            benchmark_id=result.benchmark_id,
            question=result.question,
            expected_answer=result.expected_answer,
            category=result.category,
            difficulty=result.difficulty,
            prompt_name=result.prompt_name,
            prompt_version=result.prompt_version,
            model_name=result.model_name,
            response=result.response,
            error=result.error,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
            similarity_score=result.similarity_score,
            keyword_accuracy=result.keyword_accuracy,
            overall_accuracy=result.overall_accuracy,
            hallucination_score=result.hallucination_score,
            hallucination_flags="; ".join(result.hallucination_flags),
            overall_rating=result.overall_rating,
            timestamp=result.timestamp,
        )
        session.add(record)
        session.commit()
        result.id = record.id
        return result
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.error(f"Failed to save evaluation result: {exc}")
        raise
    finally:
        session.close()


def get_all_evaluations() -> pd.DataFrame:
    """Load the full evaluation history from SQLite as a DataFrame."""
    init_db()
    session = get_session()
    try:
        records = session.query(EvaluationRecord).order_by(EvaluationRecord.timestamp.desc()).all()
        rows = [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "benchmark_id": r.benchmark_id,
                "question": r.question,
                "expected_answer": r.expected_answer,
                "category": r.category,
                "difficulty": r.difficulty,
                "prompt_name": r.prompt_name,
                "prompt_version": r.prompt_version,
                "model_name": r.model_name,
                "response": r.response,
                "error": r.error,
                "latency_ms": r.latency_ms,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "cost_usd": r.cost_usd,
                "similarity_score": r.similarity_score,
                "keyword_accuracy": r.keyword_accuracy,
                "overall_accuracy": r.overall_accuracy,
                "hallucination_score": r.hallucination_score,
                "hallucination_flags": r.hallucination_flags,
                "overall_rating": r.overall_rating,
            }
            for r in records
        ]
        return pd.DataFrame(rows)
    finally:
        session.close()


def clear_history() -> None:
    """Delete all stored evaluation results. Used by the 'Reset history' UI action."""
    init_db()
    session = get_session()
    try:
        session.query(EvaluationRecord).delete()
        session.commit()
    finally:
        session.close()


def export_csv(df: pd.DataFrame, path: str | Path = RESULTS_CSV_PATH) -> Path:
    """Export an evaluation DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Exported {len(df)} rows to {path}")
    return path


def export_json(df: pd.DataFrame, path: str | Path = REPORT_JSON_PATH) -> Path:
    """Export an evaluation DataFrame to a structured JSON report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "total_evaluations": len(df),
        "models_evaluated": sorted(df["model_name"].unique().tolist()) if not df.empty else [],
        "prompt_versions_evaluated": (
            sorted(df["prompt_version"].unique().tolist()) if not df.empty else []
        ),
        "results": df.to_dict(orient="records"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Exported JSON report with {len(df)} rows to {path}")
    return path


def build_summary(results: List[EvaluationResult]) -> pd.DataFrame:
    """Convert a list of freshly-computed EvaluationResult objects into a DataFrame."""
    return pd.DataFrame([r.to_flat_dict() for r in results])
