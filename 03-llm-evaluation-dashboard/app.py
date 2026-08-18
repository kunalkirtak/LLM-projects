"""
LLM Evaluation Dashboard
=========================

A Streamlit application for benchmarking prompts and Gemini models against
a repeatable dataset, with automated scoring, latency/cost/token tracking,
lightweight hallucination detection, and exportable evaluation reports.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from models.evaluation import PromptVersion
from utils import storage
from utils.benchmark import dataset_to_dataframe, load_benchmark
from utils.charts import (
    accuracy_by_prompt_chart,
    cost_chart,
    hallucination_trend_chart,
    latency_chart,
    token_usage_chart,
)
from utils.evaluator import LLMEvaluator

load_dotenv()

AVAILABLE_MODELS = [
    "gemini-2.5-flash"
]

DEFAULT_PROMPT_TEMPLATE = (
    "Answer the following question accurately and concisely.\n\nQuestion: {question}\nAnswer:"
)

st.set_page_config(page_title="LLM Evaluation Dashboard", page_icon="📊", layout="wide")


def _init_session_state() -> None:
    if "prompt_versions" not in st.session_state:
        st.session_state.prompt_versions = [
            PromptVersion(name="baseline", version="1.0", template=DEFAULT_PROMPT_TEMPLATE)
        ]
    if "last_run_df" not in st.session_state:
        st.session_state.last_run_df = pd.DataFrame()


def _sidebar_config() -> tuple[str, list[str], list[PromptVersion]]:
    st.sidebar.header("⚙️ Configuration")

    api_key = st.sidebar.text_input(
        "Google AI Studio API key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    selected_models = st.sidebar.multiselect(
        "Models to evaluate", options=AVAILABLE_MODELS, default=[AVAILABLE_MODELS[0]]
    )

    st.sidebar.divider()
    st.sidebar.subheader("📝 Prompt Versions")

    with st.sidebar.form("add_prompt_form", clear_on_submit=True):
        name = st.text_input("Prompt name", value="variant")
        version = st.text_input("Version", value="1.1")
        template = st.text_area(
            "Template (must include {question})", value=DEFAULT_PROMPT_TEMPLATE, height=120
        )
        submitted = st.form_submit_button("➕ Add prompt version")
        if submitted:
            try:
                new_version = PromptVersion(name=name, version=version, template=template)
                st.session_state.prompt_versions.append(new_version)
                st.sidebar.success(f"Added '{new_version.label}'")
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Invalid prompt template: {exc}")

    labels = [pv.label for pv in st.session_state.prompt_versions]
    chosen_labels = st.sidebar.multiselect("Prompt versions to run", options=labels, default=labels)
    chosen_versions = [pv for pv in st.session_state.prompt_versions if pv.label in chosen_labels]

    if st.sidebar.button("🗑️ Reset evaluation history"):
        storage.clear_history()
        st.sidebar.success("History cleared.")

    return api_key, selected_models, chosen_versions


def _run_evaluations(api_key: str, models: list[str], prompt_versions: list[PromptVersion], benchmark_items) -> pd.DataFrame:
    total_jobs = len(models) * len(prompt_versions) * len(benchmark_items)
    if total_jobs == 0:
        st.warning("Select at least one model, one prompt version, and ensure the dataset is loaded.")
        return pd.DataFrame()

    progress = st.progress(0, text="Starting evaluation...")
    results = []
    completed = 0

    for model_name in models:
        try:
            evaluator = LLMEvaluator(api_key=api_key, model_name=model_name)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not initialize model '{model_name}': {exc}")
            continue

        for prompt_version in prompt_versions:
            for item in benchmark_items:
                result = evaluator.evaluate(item, prompt_version)
                storage.save_evaluation(result)
                results.append(result)

                completed += 1
                progress.progress(
                    completed / total_jobs,
                    text=f"Evaluating... {completed}/{total_jobs} "
                    f"(model={model_name}, prompt={prompt_version.label})",
                )

    progress.empty()
    return storage.build_summary(results)


def _render_metric_cards(df: pd.DataFrame) -> None:
    cols = st.columns(6)
    if df.empty:
        for col, label in zip(cols, ["Overall", "Accuracy", "Hallucination", "Latency", "Tokens", "Cost"]):
            col.metric(label, "—")
        return

    cols[0].metric("Avg Overall Accuracy", f"{df['overall_accuracy'].mean():.1f}%")
    cols[1].metric("Avg Keyword Accuracy", f"{df['keyword_accuracy'].mean():.1f}%")
    cols[2].metric("Avg Hallucination Score", f"{df['hallucination_score'].mean():.1f}")
    cols[3].metric("Avg Latency", f"{df['latency_ms'].mean():.0f} ms")
    cols[4].metric("Total Tokens", f"{int(df['total_tokens'].sum()):,}")
    cols[5].metric("Total Cost", f"${df['cost_usd'].sum():.4f}")


def main() -> None:
    _init_session_state()

    st.title("📊 LLM Evaluation Dashboard")
    st.caption("Benchmark prompts and Gemini models with repeatable datasets and automated scoring.")

    try:
        benchmark_items = load_benchmark()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load benchmark dataset: {exc}")
        return

    api_key, selected_models, prompt_versions = _sidebar_config()

    tab_run, tab_dashboard, tab_compare, tab_history, tab_dataset = st.tabs(
        ["▶️ Run Evaluation", "📈 Dashboard", "🔍 Side-by-Side", "🕘 History", "📚 Dataset"]
    )

    with tab_run:
        st.subheader("Run a new evaluation sweep")
        st.write(
            f"This will run **{len(selected_models)} model(s)** × "
            f"**{len(prompt_versions)} prompt version(s)** × "
            f"**{len(benchmark_items)} benchmark question(s)**."
        )
        if st.button("🚀 Run Evaluation", type="primary", disabled=not api_key):
            if not api_key:
                st.error("Please provide a Google AI Studio API key in the sidebar.")
            else:
                with st.spinner("Running benchmark sweep..."):
                    run_df = _run_evaluations(api_key, selected_models, prompt_versions, benchmark_items)
                st.session_state.last_run_df = run_df
                if not run_df.empty:
                    st.success(f"Completed {len(run_df)} evaluations.")
        if not api_key:
            st.info("Add your free Gemini API key in the sidebar to enable evaluation runs.")

        if not st.session_state.last_run_df.empty:
            st.subheader("Latest run results")
            st.dataframe(st.session_state.last_run_df, use_container_width=True, height=320)

    history_df = storage.get_all_evaluations()

    with tab_dashboard:
        st.subheader("Overview")
        _render_metric_cards(history_df)

        col1, col2 = st.columns(2)
        col1.plotly_chart(latency_chart(history_df), use_container_width=True)
        col2.plotly_chart(cost_chart(history_df), use_container_width=True)

        col3, col4 = st.columns(2)
        col3.plotly_chart(token_usage_chart(history_df), use_container_width=True)
        col4.plotly_chart(accuracy_by_prompt_chart(history_df), use_container_width=True)

        st.plotly_chart(hallucination_trend_chart(history_df), use_container_width=True)

        if not history_df.empty:
            flagged = history_df[history_df["hallucination_score"] >= 40]
            if not flagged.empty:
                st.subheader("⚠️ Flagged responses (hallucination score ≥ 40)")
                st.dataframe(
                    flagged[
                        ["timestamp", "model_name", "prompt_version", "question", "hallucination_score", "hallucination_flags"]
                    ],
                    use_container_width=True,
                )

    with tab_compare:
        st.subheader("Side-by-side response comparison")
        if history_df.empty:
            st.info("Run an evaluation first to compare responses.")
        else:
            question_options = sorted(history_df["question"].unique())
            selected_question = st.selectbox("Choose a benchmark question", question_options)
            subset = history_df[history_df["question"] == selected_question]

            cols = st.columns(min(3, max(1, len(subset))))
            for i, (_, row) in enumerate(subset.iterrows()):
                col = cols[i % len(cols)]
                with col:
                    st.markdown(f"**{row['model_name']}** · prompt `{row['prompt_version']}`")
                    st.text_area("Response", row["response"], height=150, key=f"resp_{row['id']}")
                    st.caption(
                        f"Accuracy: {row['overall_accuracy']:.1f}% | "
                        f"Hallucination: {row['hallucination_score']:.1f} | "
                        f"Latency: {row['latency_ms']:.0f} ms | "
                        f"Cost: ${row['cost_usd']:.4f}"
                    )

    with tab_history:
        st.subheader("Evaluation history")
        if history_df.empty:
            st.info("No evaluations recorded yet.")
        else:
            st.dataframe(history_df, use_container_width=True, height=400)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬇️ Export CSV"):
                    path = storage.export_csv(history_df)
                    st.success(f"Exported to {path}")
            with col2:
                if st.button("⬇️ Export JSON report"):
                    path = storage.export_json(history_df)
                    st.success(f"Exported to {path}")

    with tab_dataset:
        st.subheader("Benchmark dataset")
        st.dataframe(dataset_to_dataframe(benchmark_items), use_container_width=True, height=400)


if __name__ == "__main__":
    logger.add("results/app.log", rotation="1 MB", retention=3)
    main()
