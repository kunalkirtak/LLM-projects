# 🧪 LLM Playground & Prompt Laboratory

A Streamlit workbench for experimenting with prompts, tracking token usage and cost,
and comparing LLM responses side by side — built to demonstrate practical prompt
engineering and LLM API integration skills.

---

## Overview

Working with large language models in production isn't just about writing a prompt and
reading the output — it's about understanding what a prompt *costs*, how *fast* it
returns, how it behaves under different sampling settings, and how to keep a record of
what was tried and what worked. This project packages that workflow into a single,
self-contained tool.

It was built as a portfolio project to showcase:

- Practical, hands-on **prompt engineering** (system prompts, temperature/top-p tuning)
- **LLM API integration** using the official OpenAI Python SDK
- **Token accounting and cost estimation** using `tiktoken`
- A **modular, provider-agnostic architecture** designed for easy extension
- Clean **local persistence and data export** (JSON/CSV)

---

## Features

- 📝 Prompt input area with an optional system prompt
- 🎛️ Model selector, temperature slider, top-p slider, max-tokens input
- 🚀 One-click response generation with a loading spinner and error handling
- 📊 Live metrics: input tokens, output tokens, total tokens, estimated cost, latency
- 📜 Prompt version history — save and reload any previous prompt
- 🔬 Side-by-side comparison view for two independent configurations
- 📤 Export session history to JSON or CSV
- 🧩 Clean, modular codebase designed for adding new LLM providers with minimal changes

---

## Architecture

```
┌─────────────────┐
│     app.py       │  Streamlit UI — layout, session state, user interaction
└────────┬─────────┘
         │
 ┌───────┼────────────┬──────────────┬───────────────┐
 │                     │              │               │
 ▼                     ▼              ▼               ▼
utils/llm.py     utils/metrics.py  utils/pricing.py  utils/storage.py
Provider calls   Token counting &  Per-model cost    History persistence
& result shaping latency timing    calculation       & JSON/CSV export
```

**Design principle:** `app.py` never talks to an LLM provider's SDK directly. It only
calls `generate_completion()` in `utils/llm.py` and works with the plain
`CompletionResult` object it returns. Every provider-specific detail — request shape,
response parsing, SDK client — is isolated in that one module.

**Adding a new provider (e.g. Gemini or Anthropic) only requires:**
1. Writing a `_call_<provider>()` function in `utils/llm.py`.
2. Adding one branch to `generate_completion()`.

No other file in the project needs to change.

---

## Folder Structure

```
llm-playground-lab/
├── app.py                        # Streamlit application entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── README.md
├── utils/
│   ├── __init__.py
│   ├── llm.py                    # Provider abstraction (OpenAI today, extensible)
│   ├── metrics.py                # Token counting & latency measurement
│   ├── pricing.py                # Model pricing table & cost calculation
│   └── storage.py                # Prompt history persistence & export
├── prompts/
│   └── prompt_history.json       # Local prompt/response history
├── exports/                      # Generated JSON/CSV exports land here
├── assets/                       # Screenshots / static assets for docs
└── notebooks/
    └── Prompt_Laboratory.ipynb   # Standalone notebook walkthrough
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/llm-playground-lab.git
cd llm-playground-lab

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Copy the example file and add your own API key:

```bash
cp .env.example .env
```

| Variable          | Description                              | Required |
|-------------------|-------------------------------------------|----------|
| `OPENAI_API_KEY`  | Your OpenAI API key                       | Yes      |
| `GEMINI_API_KEY`  | Reserved for future Gemini support        | No       |
| `ANTHROPIC_API_KEY` | Reserved for future Anthropic support   | No       |

---

## Running the Project

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Running the notebook:**

```bash
jupyter notebook notebooks/Prompt_Laboratory.ipynb
```

or open it directly in Google Colab.

---

## Screenshots

> Add screenshots of the Playground, Compare, and Export tabs here once the app is
> running locally.

```
assets/playground.png
assets/compare.png
assets/export.png
```

---

## Demo

> Add a link to a hosted demo (Streamlit Community Cloud) or a short screen recording
> here once deployed.

---

## Future Improvements

- [ ] Add Gemini and Anthropic providers alongside OpenAI
- [ ] Streaming token-by-token response rendering
- [ ] Prompt templates library with variable substitution
- [ ] Automatic prompt scoring / evaluation harness
- [ ] Multi-turn conversation mode with message history
- [ ] Dockerfile for one-command deployment
- [ ] Unit test suite (pytest) covering `utils/`

---

## Technologies Used

- **Python 3.11+**
- **Streamlit** — interactive UI
- **OpenAI SDK** — LLM API integration
- **tiktoken** — accurate token counting
- **pandas** — tabular history handling
- **matplotlib** — latency/temperature visualization (notebook)
- **python-dotenv** — environment variable management

---

## Skills Demonstrated

- Prompt engineering (system prompts, temperature/top-p tuning, prompt versioning)
- LLM API integration and error handling (auth, rate limits, provider errors)
- Token-level cost accounting and budgeting
- Modular Python architecture designed for extensibility
- Local data persistence and export pipelines (JSON/CSV)
- Building usable internal tooling with Streamlit

---

## Learning Outcomes

Building this project involved reasoning about the parts of LLM application
development that don't show up in a basic API tutorial: how token usage translates
to real cost, how sampling parameters affect output variability, how to structure code
so a second model provider doesn't require a rewrite, and how to design a small tool
that's actually pleasant to use for repeated experimentation.

---

## License

Licensed under the [MIT License](https://opensource.org/licenses/MIT).
