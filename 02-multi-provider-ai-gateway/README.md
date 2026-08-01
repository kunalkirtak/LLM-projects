# Multi-Provider AI Gateway

A production-inspired FastAPI backend that exposes a single, unified `/chat` API for
OpenAI, Anthropic (Claude), and Google Gemini — with automatic retries, provider
fallback, streaming, structured JSON outputs, and built-in cost/latency/token metrics.

Think of it as a thin, well-engineered abstraction layer sitting in front of three
different LLM SDKs: callers send one request shape and get one response shape back,
regardless of which vendor actually served it.

---

## Features

- **Unified `/chat` endpoint** — one request/response contract for three providers
- **Provider abstraction layer** — every provider implements the same `BaseProvider`
  interface; adding a new one touches two files
- **Automatic fallback** — if the requested provider fails, the gateway walks a
  configurable fallback chain until one succeeds
- **Exponential backoff retries** — transient errors (rate limits, timeouts, 5xx) are
  retried automatically via `tenacity`; auth errors fail fast
- **Streaming responses** — Server-Sent Events (SSE) for token-by-token output
- **Structured JSON outputs** — native JSON mode where supported, prompt-level
  enforcement where it isn't
- **Token usage & cost tracking** — every response reports input/output/total tokens
  and an estimated USD cost
- **Latency measurement** — per-request and aggregate average latency
- **Centralized metrics** — `/metrics` exposes running totals per provider
- **Structured request logging** — every request is logged to a rotating file with
  timestamp, provider, tokens, cost, latency, and status
- **Health check** — `/health` reports which providers are actually configured
- **Pydantic v2 validation** — every request and response is a typed, validated model

---

## Architecture

```
                    ┌───────────────────────┐
                    │        Client         │
                    └───────────┬───────────┘
                                │  POST /chat
                                ▼
                    ┌───────────────────────┐
                    │   FastAPI (main.py)   │
                    │   - CORS, timing,     │
                    │     error handling    │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Routes (routes.py)  │
                    │   - request validation│
                    │   - streaming vs sync │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Gateway (gateway.py)│
                    │   - provider routing  │
                    │   - retry (tenacity)  │
                    │   - fallback chain    │
                    │   - metrics + logging │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │ OpenAIProvider  │ │AnthropicProvider│ │ GeminiProvider │
     │ (base.py iface) │ │ (base.py iface) │ │ (base.py iface)│
     └────────┬────────┘ └────────┬────────┘ └────────┬───────┘
              ▼                   ▼                    ▼
        OpenAI API          Anthropic API         Gemini API
```

Every provider adapter implements the same `BaseProvider` interface
(`complete()` / `stream()` / `is_configured()`), so the gateway never
imports a vendor SDK directly — it only depends on the interface. This
is what makes fallback, retries, and metrics vendor-agnostic.

---

## Folder Structure

```
multi-provider-ai-gateway/
├── app/
│   ├── main.py                  # FastAPI app, middleware, startup
│   ├── api/
│   │   └── routes.py            # /chat, /health, /providers, /metrics
│   ├── providers/
│   │   ├── base.py              # BaseProvider interface + exceptions
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── gemini_provider.py
│   │   └── gateway.py           # routing, retry, fallback orchestration
│   ├── models/
│   │   ├── request.py           # Pydantic request schemas
│   │   └── response.py          # Pydantic response schemas
│   └── utils/
│       ├── config.py            # env-driven settings (pydantic-settings)
│       ├── logging.py           # Loguru setup + structured request logging
│       ├── pricing.py           # per-model USD/1M-token pricing table
│       └── metrics.py           # in-process metrics aggregation
├── logs/
│   └── requests.log             # rotating structured request log
├── notebooks/
│   └── Gateway_Demo.ipynb       # end-to-end demo notebook
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/multi-provider-ai-gateway.git
cd multi-provider-ai-gateway

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your API keys
```

You don't need all three provider keys — the gateway works with just one
configured; fallback simply skips any provider that isn't configured.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `OPENAI_DEFAULT_MODEL` | Default OpenAI model | `gpt-4o-mini` |
| `ANTHROPIC_DEFAULT_MODEL` | Default Anthropic model | `claude-3-5-sonnet-20241022` |
| `GEMINI_DEFAULT_MODEL` | Default Gemini model | `gemini-1.5-flash` |
| `DEFAULT_PROVIDER` | Provider used when the request omits one | `openai` |
| `FALLBACK_ORDER` | Comma-separated fallback order | `openai,anthropic,gemini` |
| `ENABLE_FALLBACK` | Master switch for fallback behavior | `true` |
| `MAX_RETRIES` | Retry attempts per provider before failing over | `3` |
| `RETRY_MIN_WAIT_SECONDS` / `RETRY_MAX_WAIT_SECONDS` | Exponential backoff bounds | `1.0` / `10.0` |
| `REQUEST_TIMEOUT_SECONDS` | Per-request timeout | `60.0` |
| `LOG_LEVEL` | Loguru log level | `INFO` |
| `LOG_FILE_PATH` | Path to the rotating request log | `logs/requests.log` |
| `HOST` / `PORT` | Uvicorn bind address | `0.0.0.0` / `8000` |

---

## Running Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs (Swagger UI): `http://localhost:8000/docs`
Alternative docs (ReDoc): `http://localhost:8000/redoc`

---

## API Documentation

### `POST /chat`
Send a chat completion request. Set `"stream": true` to receive an
SSE stream instead of a single JSON response.

### `GET /health`
Returns overall service status and, per provider, whether it has valid
credentials configured.

### `GET /providers`
Lists every registered provider along with its default model and
capabilities (streaming, JSON mode).

### `GET /metrics`
Returns aggregated usage metrics (requests, tokens, cost, latency,
retries, fallbacks) broken down by provider.

---

## Example Requests

**Basic chat completion**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Explain vector databases in two sentences."}
    ],
    "temperature": 0.7,
    "max_tokens": 300
  }'
```

**Streaming**
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "messages": [{"role": "user", "content": "Write a haiku about backends."}],
    "stream": true
  }'
```

**Structured JSON output**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "messages": [{"role": "user", "content": "List 3 REST API best practices as JSON."}],
    "response_format": {"type": "json_object"}
  }'
```

---

## Example Responses

**`POST /chat`**
```json
{
  "request_id": "8f14e45f-ceea-4b3a-8d17-3b1f0c2a9e21",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "content": "Vector databases store data as high-dimensional embeddings...",
  "finish_reason": "stop",
  "usage": {
    "input_tokens": 24,
    "output_tokens": 58,
    "total_tokens": 82,
    "estimated_cost_usd": 0.0000387,
    "latency_ms": 812.44,
    "retry_count": 0,
    "fallback_used": false
  },
  "fallback_chain_attempted": []
}
```

**`GET /health`**
```json
{
  "status": "healthy",
  "providers": [
    {"provider": "openai", "configured": true, "status": "ready"},
    {"provider": "anthropic", "configured": true, "status": "ready"},
    {"provider": "gemini", "configured": false, "status": "not_configured"}
  ],
  "version": "1.0.0"
}
```

---

## Future Improvements

- Swap in-process `MetricsStore` for Prometheus counters/histograms + Grafana dashboard
- Add Redis-backed request/response caching for idempotent prompts
- Add per-API-key rate limiting and usage quotas
- Persist structured logs to a database (Postgres/ClickHouse) instead of flat files
- Add automatic model-tier fallback (e.g., drop from GPT-4o to GPT-4o-mini on budget cap)
- Add a `/batch` endpoint for bulk async completions
- Containerize with Docker + docker-compose (app + log shipper)
- Add a test suite (pytest + httpx.AsyncClient) with mocked provider SDKs

---

## Technologies Used

- **Python 3.11+**
- **FastAPI** — async web framework
- **Uvicorn** — ASGI server
- **Pydantic v2 / pydantic-settings** — validation and config
- **OpenAI SDK**, **Anthropic SDK**, **Google Generative AI SDK**
- **httpx** — async HTTP client
- **Tenacity** — retry with exponential backoff
- **Loguru** — structured logging
- **Pandas** — used in the demo notebook for metrics analysis
- **asyncio** — concurrency throughout the provider/gateway layer

---

## Skills Demonstrated

- Designing a clean abstraction layer over multiple heterogeneous third-party SDKs
- Async Python backend architecture with FastAPI
- Resilience engineering: retries, exponential backoff, graceful degradation, fallback chains
- Structured logging and in-process metrics aggregation
- API design: request/response contracts, error envelopes, streaming (SSE)
- Pydantic-based validation and typed configuration management
- Writing modular, dependency-injectable, testable service code

---

## License

MIT License. See `LICENSE` for details.
