"""Opt-in bearer authentication at the HTTP boundary.

Keep this pure ASGI: wrapping streaming responses in a body-consuming middleware
would change cancellation/backpressure behavior in the shared generation pipeline.
Local desktop installs remain keyless unless --api-key is explicitly configured.
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class AuthenticationMiddleware:
    def __init__(self, app: ASGIApp, keys: tuple[str, ...]) -> None:
        self.app = app
        # Fixed-size digests avoid length-dependent comparison and never retain the
        # plaintext keys on the middleware. Check every configured key, even on a
        # match, so key rotation does not reveal which list entry was accepted.
        self._digests = tuple(hashlib.sha256(key.encode("utf-8")).digest() for key in keys)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._digests:
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        root_path = scope.get("root_path", "").rstrip("/")
        if root_path and path.startswith(root_path + "/"):
            path = path[len(root_path):]
        # These probes must work while the engine is loading. Admin handlers keep
        # their own socket-peer loopback check; this exemption grants no admin access.
        if path in {"/health", "/healthz", "/readyz"} or path.startswith("/v1/admin/"):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope).getlist("authorization")
        scheme, _, key = headers[0].partition(" ") if len(headers) == 1 else ("", "", "")
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        matched = False
        for expected in self._digests:
            matched |= hmac.compare_digest(digest, expected)
        if scheme.lower() == "bearer" and key and matched:
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            {"error": {
                "message": "Invalid or missing API key",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            }},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)


def install_auth(app: FastAPI, keys_csv: str = "") -> None:
    """Install before serving; empty input preserves the existing keyless behavior."""
    keys = tuple(key.strip() for key in keys_csv.split(",") if key.strip())
    if keys:
        app.add_middleware(AuthenticationMiddleware, keys=keys)
