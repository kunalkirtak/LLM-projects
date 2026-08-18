# 🧠 LLM-Projects

**A portfolio of production-style LLM engineering projects** — prompt experimentation & cost tracking, a multi-provider AI gateway, and an LLM evaluation dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LLMs](https://img.shields.io/badge/LLMs-OpenAI%20%7C%20Anthropic%20%7C%20Gemini-8A2BE2)
![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Where [RAG-projects](https://github.com/kunalkirtak/RAG-projects) focuses on retrieval, this repo focuses on the **LLM application layer itself**: prompt engineering workflow, provider abstraction/reliability, and systematic model evaluation — the tooling around calling an LLM well, not just calling one.

---

## 📁 Projects

| # | Project | What it does | Core additions |
|---|---|---|---|
| 01 | [**LLM Playground & Prompt Laboratory**](./01-llm-playground-lab) | Streamlit workbench for experimenting with prompts, comparing responses side by side, and tracking token usage/cost/latency. | OpenAI SDK integration, `tiktoken` cost accounting, prompt version history, JSON/CSV export, provider-agnostic architecture |
| 02 | [**Multi-Provider AI Gateway**](./02-multi-provider-ai-gateway) | FastAPI backend exposing one unified `/chat` API across OpenAI, Anthropic (Claude), and Google Gemini. | `BaseProvider` abstraction, automatic fallback chain, exponential-backoff retries (`tenacity`), SSE streaming, structured JSON outputs, `/metrics` + `/health` endpoints |
| 03 | [**LLM Evaluation Dashboard**](./03-llm-evaluation-dashboard) | Benchmarking platform for scoring prompts/models against a repeatable dataset, with an interactive dashboard. | TF-IDF similarity + keyword scoring, hallucination heuristic, SQLite-backed history, Plotly dashboard, CSV/JSON report export |

Every project folder has its own detailed README covering architecture, setup, and usage — click through above for the full write-up.

---

## 🧠 What this repo demonstrates

- **Prompt engineering workflow** — systematic experimentation, versioning, and side-by-side comparison rather than ad-hoc trial and error
- **Multi-vendor LLM integration** — a single clean interface (`BaseProvider`) sitting in front of the OpenAI, Anthropic, and Gemini SDKs
- **Reliability patterns** — retries with exponential backoff, provider fallback chains, structured error handling, and health checks
- **Cost & performance observability** — token accounting, per-model cost estimation, and latency tracking built into every project
- **Evaluation discipline** — repeatable benchmarking with quantitative scoring (similarity, keyword coverage) and hallucination detection, instead of eyeballing outputs
- **Productionization** — FastAPI services, Streamlit UIs, structured logging (Loguru), Pydantic v2 validation, Docker-ready layouts

## 🛠️ Common Tech Stack

| Layer | Tools used across projects |
|---|---|
| LLM providers | OpenAI, Anthropic (Claude), Google Gemini |
| Backend | FastAPI, Pydantic v2 |
| Frontend | Streamlit |
| Reliability | `tenacity` (retries), fallback chains, `loguru` structured logging |
| Cost / token tracking | `tiktoken`, per-provider pricing tables |
| Evaluation | scikit-learn (TF-IDF similarity), SQLAlchemy/SQLite, Plotly |
| Data | pandas, numpy |

## 📂 Repository Structure

```
LLM-projects/
├── 01-llm-playground-lab/          # Prompt experimentation & cost tracking workbench
├── 02-multi-provider-ai-gateway/   # Unified multi-provider LLM API gateway
├── 03-llm-evaluation-dashboard/    # Prompt/model benchmarking & evaluation dashboard
├── LICENSE
└── README.md
```

## 🚀 Getting Started

Each project is self-contained with its own dependencies and `.env` setup.

```bash
git clone https://github.com/kunalkirtak/LLM-projects.git
cd LLM-projects/<project-folder>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env          # add your provider API key(s)
```

Then follow the setup and usage instructions in that project's own `README.md` (running the Streamlit app / FastAPI server, Docker, etc.).

## 📜 License

This repository is licensed under the [MIT License](./LICENSE).

## 👤 Author

**Kunal Kirtak**
GitHub: [@kunalkirtak](https://github.com/kunalkirtak)

If you find this useful, consider ⭐ starring the repo!
