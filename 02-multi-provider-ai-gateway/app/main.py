"""
Application entrypoint.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Or simply:
    python -m app.main
"""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.utils.config import get_settings
from app.utils.logging import logger

settings = get_settings()

app = FastAPI(
    title="Multi-Provider AI Gateway",
    description=(
        "A single, unified API for chatting with OpenAI, Anthropic, and Google "
        "Gemini models -- with automatic retries, provider fallback, structured "
        "outputs, streaming, and built-in cost/latency/token metrics."
    ),
    version="1.0.0",
    contact={"name": "Multi-Provider AI Gateway"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Attach request timing to every response for basic observability."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so an unexpected error never leaks a raw traceback to clients."""
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"error_type": "InternalServerError", "message": "An unexpected error occurred."},
    )


app.include_router(router, prefix="", tags=["gateway"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Multi-Provider AI Gateway",
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Multi-Provider AI Gateway on {settings.host}:{settings.port}")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.environment == "development")
