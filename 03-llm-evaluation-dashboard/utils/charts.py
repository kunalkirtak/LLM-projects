"""
Plotly chart builders for the Streamlit dashboard.

Every function takes the evaluation results DataFrame and returns a ready-to
-render `plotly.graph_objects.Figure`. Keeping chart construction out of
app.py keeps the UI file focused on layout rather than plotting logic.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_TEMPLATE = "plotly_white"


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14))
    fig.update_layout(template=CHART_TEMPLATE, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def latency_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing average latency by model."""
    if df.empty:
        return _empty_figure("No evaluation data yet")
    grouped = df.groupby("model_name", as_index=False)["latency_ms"].mean()
    fig = px.bar(
        grouped, x="model_name", y="latency_ms", color="model_name",
        title="Average Latency by Model", labels={"latency_ms": "Latency (ms)", "model_name": "Model"},
    )
    fig.update_layout(template=CHART_TEMPLATE, showlegend=False)
    return fig


def cost_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing total estimated cost by model."""
    if df.empty:
        return _empty_figure("No evaluation data yet")
    grouped = df.groupby("model_name", as_index=False)["cost_usd"].sum()
    fig = px.bar(
        grouped, x="model_name", y="cost_usd", color="model_name",
        title="Total Estimated Cost by Model", labels={"cost_usd": "Cost (USD)", "model_name": "Model"},
    )
    fig.update_layout(template=CHART_TEMPLATE, showlegend=False)
    return fig


def token_usage_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of input vs. output token usage per model."""
    if df.empty:
        return _empty_figure("No evaluation data yet")
    grouped = df.groupby("model_name", as_index=False)[["input_tokens", "output_tokens"]].sum()
    melted = grouped.melt(id_vars="model_name", var_name="token_type", value_name="tokens")
    fig = px.bar(
        melted, x="model_name", y="tokens", color="token_type", barmode="stack",
        title="Token Usage by Model", labels={"tokens": "Tokens", "model_name": "Model"},
    )
    fig.update_layout(template=CHART_TEMPLATE)
    return fig


def accuracy_by_prompt_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of overall accuracy per prompt version, split by model."""
    if df.empty:
        return _empty_figure("No evaluation data yet")
    grouped = df.groupby(["prompt_version", "model_name"], as_index=False)["overall_accuracy"].mean()
    fig = px.bar(
        grouped, x="prompt_version", y="overall_accuracy", color="model_name", barmode="group",
        title="Accuracy by Prompt Version",
        labels={"overall_accuracy": "Overall Accuracy (%)", "prompt_version": "Prompt Version"},
    )
    fig.update_layout(template=CHART_TEMPLATE)
    return fig


def hallucination_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of hallucination score over time (evaluation order)."""
    if df.empty:
        return _empty_figure("No evaluation data yet")
    sorted_df = df.sort_values("timestamp")
    fig = px.line(
        sorted_df, x="timestamp", y="hallucination_score", color="model_name", markers=True,
        title="Hallucination Score Trend",
        labels={"hallucination_score": "Hallucination Score", "timestamp": "Time"},
    )
    fig.update_layout(template=CHART_TEMPLATE)
    return fig
