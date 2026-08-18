"""
Pydantic data models shared across the evaluation pipeline.

These models are the single source of truth for the shape of a benchmark
item and an evaluation result. Keeping them centralized avoids the
"dict soup" problem that tends to creep into evaluation scripts.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BenchmarkItem(BaseModel):
    """A single row of the benchmark dataset."""

    id: int
    question: str
    expected_answer: str
    category: str
    difficulty: str
    reference_keywords: List[str] = Field(default_factory=list)

    @field_validator("difficulty")
    @classmethod
    def normalize_difficulty(cls, value: str) -> str:
        return value.strip().title()


class PromptVersion(BaseModel):
    """A named, versioned prompt template.

    The template must contain a ``{question}`` placeholder that gets
    filled in with the benchmark question at evaluation time.
    """

    name: str
    version: str
    template: str

    @field_validator("template")
    @classmethod
    def must_contain_placeholder(cls, value: str) -> str:
        if "{question}" not in value:
            raise ValueError("Prompt template must contain a {question} placeholder")
        return value

    @property
    def label(self) -> str:
        return f"{self.name} (v{self.version})"


class HallucinationReport(BaseModel):
    """Result of the hallucination heuristic for a single response."""

    score: float = Field(ge=0, le=100, description="0 = fully grounded, 100 = fully hallucinated")
    missing_keywords: List[str] = Field(default_factory=list)
    unsupported_statements: List[str] = Field(default_factory=list)
    explanations: List[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """The full record produced by evaluating one (question, prompt, model) triple."""

    id: Optional[int] = None
    benchmark_id: int
    question: str
    expected_answer: str
    category: str
    difficulty: str

    prompt_name: str
    prompt_version: str
    model_name: str

    response: str
    error: Optional[str] = None

    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float

    similarity_score: float = Field(ge=0, le=100)
    keyword_accuracy: float = Field(ge=0, le=100)
    overall_accuracy: float = Field(ge=0, le=100)

    hallucination_score: float = Field(ge=0, le=100)
    hallucination_flags: List[str] = Field(default_factory=list)

    overall_rating: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_flat_dict(self) -> dict:
        """Flatten nested list fields into strings for CSV-friendly export."""
        data = self.model_dump()
        data["hallucination_flags"] = "; ".join(self.hallucination_flags)
        data["timestamp"] = self.timestamp.isoformat()
        return data
