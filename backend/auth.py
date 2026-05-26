from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config import API_KEY

# Endpoints that don't require auth (health check, metrics, static files)
_PUBLIC_PATHS = {"/health", "/metrics"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Checks X-API-Key header against the configured API_KEY.

    If API_KEY is not set (empty/None), all requests pass through — this
    keeps local development frictionless while securing deployed instances.
    """

    async def dispatch(self, request: Request, call_next):
        if not API_KEY:
            return await call_next(request)

        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Allow static file serving without auth
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Неверный или отсутствующий API-ключ"},
            )

        return await call_next(request)
