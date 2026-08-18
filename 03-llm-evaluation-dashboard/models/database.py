"""
SQLAlchemy persistence layer for evaluation runs.

We use SQLite because the dashboard is meant to run locally / in a single
Streamlit process, so there is no need for a networked database. The engine
path is read from DATABASE_URL so it can be swapped later without touching
application code.
"""

from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///results/evaluations.db")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class EvaluationRecord(Base):
    """ORM table storing every evaluation run for historical analysis."""

    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(Integer, nullable=False)
    question = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=False)
    category = Column(String(64), nullable=False)
    difficulty = Column(String(32), nullable=False)

    prompt_name = Column(String(128), nullable=False)
    prompt_version = Column(String(32), nullable=False)
    model_name = Column(String(64), nullable=False)

    response = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    latency_ms = Column(Float, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)

    similarity_score = Column(Float, nullable=False)
    keyword_accuracy = Column(Float, nullable=False)
    overall_accuracy = Column(Float, nullable=False)

    hallucination_score = Column(Float, nullable=False)
    hallucination_flags = Column(Text, nullable=True)

    overall_rating = Column(String(32), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


_engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """Create all tables if they do not already exist. Safe to call repeatedly."""
    os.makedirs("results", exist_ok=True)
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    return SessionLocal()
