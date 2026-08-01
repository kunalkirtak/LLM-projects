"""
storage.py
----------
Local persistence for prompt history, plus JSON/CSV export helpers.

History is stored as a flat JSON array on disk. For a project of this
scope a JSON file is the right tool -- it needs no external database,
it's human-readable, and it's trivial to inspect or version-control.
"""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = Path(__file__).resolve().parent.parent / "prompts" / "prompt_history.json"
EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports"


@dataclass
class PromptRecord:
    """A single saved prompt/response run, ready for JSON serialization."""

    timestamp: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    system_prompt: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    latency_seconds: float


def _ensure_history_file() -> None:
    """Create the history file with an empty array if it doesn't exist."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_history() -> list[dict]:
    """
    Load all saved prompt records.

    Returns:
        List of record dicts, newest last. Returns an empty list if the
        history file is missing or corrupted, rather than raising, since
        a broken history file shouldn't block the rest of the app.
    """
    _ensure_history_file()
    try:
        raw = HISTORY_FILE.read_text(encoding="utf-8")
        return json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError:
        return []


def save_prompt_record(record: PromptRecord) -> None:
    """Append a new record to the local history file."""
    _ensure_history_file()
    history = load_history()
    history.append(asdict(record))
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def build_record(
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    system_prompt: str,
    prompt: str,
    response: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost: float,
    latency_seconds: float,
) -> PromptRecord:
    """Construct a PromptRecord with the current UTC timestamp attached."""
    return PromptRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        prompt=prompt,
        response=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost=estimated_cost,
        latency_seconds=latency_seconds,
    )


def export_to_json(records: list[dict], filename: str = "session_export.json") -> Path:
    """
    Write a list of records to a JSON file in the exports directory.

    Returns:
        Path to the written file.
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXPORTS_DIR / filename
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_path


def export_to_csv(records: list[dict], filename: str = "session_export.csv") -> Path:
    """
    Write a list of records to a CSV file in the exports directory.

    Returns:
        Path to the written file. Writes only the header row if records
        is empty, so downstream tools never see a malformed file.
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXPORTS_DIR / filename

    fieldnames = list(PromptRecord.__annotations__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    return output_path
