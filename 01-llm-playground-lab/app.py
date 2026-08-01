"""
app.py
------
LLM Playground & Prompt Laboratory
===================================
A Streamlit workbench for experimenting with prompts, comparing model
outputs side by side, and tracking token usage / cost / latency across
runs.

Run with:
    streamlit run app.py
"""

import streamlit as st

from utils.llm import LLMRequestError, generate_completion
from utils.pricing import available_models
from utils.storage import build_record, export_to_csv, export_to_json, load_history, save_prompt_record

st.set_page_config(
    page_title="LLM Playground & Prompt Laboratory",
    page_icon="🧪",
    layout="wide",
)


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
# Streamlit reruns the whole script on every interaction, so anything that
# needs to survive a rerun (results from previous generations, the prompt
# text a user is editing, etc.) is kept in st.session_state.
def init_session_state() -> None:
    defaults = {
        "last_result": None,        # most recent CompletionResult
        "compare_result_a": None,   # left side of comparison view
        "compare_result_b": None,   # right side of comparison view
        "prompt_text": "",
        "system_prompt_text": "",
        "session_records": [],      # records generated in this session, for export
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --------------------------------------------------------------------------
# Sidebar -- model & generation configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Generation Settings")

    model = st.selectbox("Model", options=available_models(), index=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    max_tokens = st.number_input("Max Tokens", min_value=16, max_value=4096, value=512, step=16)

    st.divider()
    st.header("📜 Prompt History")

    history = load_history()
    if history:
        # Show newest first; label with timestamp + truncated prompt so
        # entries are distinguishable at a glance.
        labels = [
            f"{record['timestamp'][:19]} — {record['prompt'][:40]}"
            for record in reversed(history)
        ]
        selected_label = st.selectbox("Load a previous prompt", options=["—"] + labels)
        if selected_label != "—":
            index_from_end = labels.index(selected_label)
            selected_record = list(reversed(history))[index_from_end]
            if st.button("Load into editor"):
                st.session_state["prompt_text"] = selected_record["prompt"]
                st.session_state["system_prompt_text"] = selected_record["system_prompt"]
                st.rerun()
    else:
        st.caption("No saved prompts yet. Generate a response and save it to build history.")


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("🧪 LLM Playground & Prompt Laboratory")
st.caption("Experiment with prompts. Track tokens, cost, and latency. Compare responses side by side.")

tab_playground, tab_compare, tab_export = st.tabs(["▶️ Playground", "🔬 Compare", "📤 Export"])


# --------------------------------------------------------------------------
# Tab 1: Playground -- single prompt in, single response out
# --------------------------------------------------------------------------
with tab_playground:
    col_input, col_output = st.columns(2)

    with col_input:
        st.subheader("Prompt")
        system_prompt = st.text_area(
            "System Prompt (optional)",
            value=st.session_state["system_prompt_text"],
            height=100,
            placeholder="You are a helpful assistant that answers concisely.",
            key="system_prompt_input",
        )
        user_prompt = st.text_area(
            "User Prompt",
            value=st.session_state["prompt_text"],
            height=220,
            placeholder="Ask anything...",
            key="user_prompt_input",
        )

        generate_clicked = st.button("🚀 Generate Response", type="primary", use_container_width=True)

    with col_output:
        st.subheader("Response")
        response_placeholder = st.empty()

        if generate_clicked:
            with st.spinner("Generating response..."):
                try:
                    result = generate_completion(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                    st.session_state["last_result"] = result
                    st.session_state["prompt_text"] = user_prompt
                    st.session_state["system_prompt_text"] = system_prompt
                except LLMRequestError as error:
                    st.error(f"⚠️ {error}")
                    st.session_state["last_result"] = None

        if st.session_state["last_result"] is not None:
            response_placeholder.markdown(st.session_state["last_result"].text)
        else:
            response_placeholder.info("Your generated response will appear here.")

    # ---- Metrics section -------------------------------------------------
    result = st.session_state["last_result"]
    if result is not None:
        st.divider()
        st.subheader("📊 Metrics")

        metric_cols = st.columns(5)
        metric_cols[0].metric("Input Tokens", result.token_usage.input_tokens)
        metric_cols[1].metric("Output Tokens", result.token_usage.output_tokens)
        metric_cols[2].metric("Total Tokens", result.token_usage.total_tokens)
        metric_cols[3].metric("Estimated Cost", f"${result.estimated_cost:.6f}")
        metric_cols[4].metric("Latency", f"{result.latency_seconds:.2f}s")

        if st.button("💾 Save Prompt to History"):
            record = build_record(
                model=result.model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                system_prompt=st.session_state["system_prompt_text"],
                prompt=st.session_state["prompt_text"],
                response=result.text,
                input_tokens=result.token_usage.input_tokens,
                output_tokens=result.token_usage.output_tokens,
                estimated_cost=result.estimated_cost,
                latency_seconds=result.latency_seconds,
            )
            save_prompt_record(record)
            st.session_state["session_records"].append(record.__dict__)
            st.success("Saved to prompt history.")


# --------------------------------------------------------------------------
# Tab 2: Compare -- run the same or different prompts side by side
# --------------------------------------------------------------------------
with tab_compare:
    st.caption("Run two configurations side by side to compare quality, cost, and latency.")

    col_a, col_b = st.columns(2)

    def render_compare_column(label: str, state_key: str) -> None:
        """Render one half of the comparison view (used for both A and B)."""
        st.markdown(f"### {label}")
        col_model = st.selectbox("Model", options=available_models(), key=f"{state_key}_model")
        col_temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05, key=f"{state_key}_temp")
        col_prompt = st.text_area("Prompt", height=150, key=f"{state_key}_prompt")

        if st.button(f"Generate {label}", key=f"{state_key}_button"):
            with st.spinner("Generating..."):
                try:
                    st.session_state[state_key] = generate_completion(
                        system_prompt="",
                        user_prompt=col_prompt,
                        model=col_model,
                        temperature=col_temp,
                        top_p=1.0,
                        max_tokens=max_tokens,
                    )
                except LLMRequestError as error:
                    st.error(f"⚠️ {error}")

        result = st.session_state[state_key]
        if result is not None:
            st.markdown(result.text)
            m1, m2, m3 = st.columns(3)
            m1.metric("Tokens", result.token_usage.total_tokens)
            m2.metric("Cost", f"${result.estimated_cost:.6f}")
            m3.metric("Latency", f"{result.latency_seconds:.2f}s")

    with col_a:
        render_compare_column("Response A", "compare_result_a")
    with col_b:
        render_compare_column("Response B", "compare_result_b")


# --------------------------------------------------------------------------
# Tab 3: Export -- dump session records to JSON / CSV
# --------------------------------------------------------------------------
with tab_export:
    st.caption("Export everything saved to history during this session.")

    all_history = load_history()
    st.write(f"**{len(all_history)}** record(s) currently saved to local history.")

    col_json, col_csv = st.columns(2)
    with col_json:
        if st.button("Export to JSON", use_container_width=True):
            path = export_to_json(all_history)
            st.success(f"Exported to `{path}`")
            with open(path, "rb") as file_handle:
                st.download_button("⬇️ Download JSON", file_handle, file_name=path.name)

    with col_csv:
        if st.button("Export to CSV", use_container_width=True):
            path = export_to_csv(all_history)
            st.success(f"Exported to `{path}`")
            with open(path, "rb") as file_handle:
                st.download_button("⬇️ Download CSV", file_handle, file_name=path.name)

    if all_history:
        st.divider()
        st.subheader("History Preview")
        st.dataframe(all_history, use_container_width=True)
