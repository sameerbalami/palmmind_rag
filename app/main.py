import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors

from app.api.deps import get_vector_store
from app.api.routes import bookings, chat, documents
from app.core.config import get_settings
from app.db.session import init_db
from app.services.llm import LLMError, RateLimitError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    get_vector_store().ensure_collection()
    yield


app = FastAPI(title=get_settings().app_name, version="1.0.0", lifespan=lifespan)

API_PREFIX = "/api/v1"
app.include_router(documents.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(bookings.router, prefix=API_PREFIX)


@app.exception_handler(RateLimitError)
async def rate_limit_handler(_: Request, exc: RateLimitError) -> JSONResponse:
    retry_after = math.ceil(exc.retry_after_seconds)
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Embedding quota reached on the LLM provider. Retry in {retry_after} seconds."},
        headers={"Retry-After": str(retry_after)},
    )


@app.exception_handler(LLMError)
@app.exception_handler(genai_errors.APIError)
async def llm_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("LLM provider error")
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": f"LLM provider error: {exc}"},
    )


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
