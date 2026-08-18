"""
Automatic scoring pipeline: response similarity and keyword-based accuracy.

Similarity is computed with a TF-IDF vectorizer + cosine similarity rather
than pulling in a heavy embedding model, which keeps the project dependency
footprint light and fully offline-capable for scoring (only generation
requires the network).
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean(text: str) -> str:
    """Lowercase and strip punctuation for more forgiving text comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity_score(response: str, expected_answer: str) -> float:
    """Score how semantically close a response is to the expected answer.

    Uses TF-IDF cosine similarity as a lightweight proxy for semantic
    similarity. Returns a value on a 0-100 scale.

    Args:
        response: The model-generated response.
        expected_answer: The reference/expected answer from the benchmark.

    Returns:
        Similarity score between 0 and 100.
    """
    resp_clean = _clean(response)
    exp_clean = _clean(expected_answer)

    if not resp_clean or not exp_clean:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([resp_clean, exp_clean])
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    except ValueError:
        # Happens when both strings are made up entirely of stopwords.
        return 0.0

    return float(np.clip(score * 100, 0, 100))


def keyword_accuracy(response: str, reference_keywords: List[str]) -> float:
    """Score the fraction of required reference keywords present in the response.

    Args:
        response: The model-generated response.
        reference_keywords: Keywords the response is expected to mention.

    Returns:
        Percentage (0-100) of reference keywords found in the response.
    """
    if not reference_keywords:
        return 100.0  # nothing required -> nothing to fail

    resp_clean = _clean(response)
    found = sum(1 for kw in reference_keywords if _clean(kw) in resp_clean)
    return round((found / len(reference_keywords)) * 100, 2)


def missing_keywords(response: str, reference_keywords: List[str]) -> List[str]:
    """Return the subset of reference keywords not found in the response."""
    resp_clean = _clean(response)
    return [kw for kw in reference_keywords if _clean(kw) not in resp_clean]


def overall_accuracy(similarity: float, keyword_acc: float, weight_similarity: float = 0.5) -> float:
    """Blend similarity and keyword accuracy into a single accuracy score.

    Args:
        similarity: Similarity score (0-100).
        keyword_acc: Keyword accuracy score (0-100).
        weight_similarity: Weight given to similarity vs. keyword accuracy.

    Returns:
        Weighted overall accuracy score (0-100).
    """
    weight_similarity = float(np.clip(weight_similarity, 0, 1))
    blended = similarity * weight_similarity + keyword_acc * (1 - weight_similarity)
    return round(blended, 2)


def rating_from_score(score: float) -> str:
    """Map a numeric overall accuracy score to a human-readable rating band."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"
