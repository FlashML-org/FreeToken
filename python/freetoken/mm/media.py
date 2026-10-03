"""Client-supplied image handling: ref collection, the accept gate, and byte fetching.

Raises plain ValueError on bad input; the server layer maps it to its wire error type.
The message names the class of failure only: no local path, allowlist root, errno or
library text reaches the client (that detail goes to the server log).
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.server.args import ServerArgs

logger = init_logger(__name__)

_MAX_IMAGE_BYTES = 32 << 20
_MAX_REDIRECTS = 5


def collect_image_refs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pop the image refs out of rendered messages in prompt order; the bare {"type": "image"} parts stay for the chat template."""
    refs: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                ref = part.pop("freetoken_ref", None)
                if ref:
                    refs.append(ref)
    return refs


def image_reject_reason(config: ServerArgs) -> str | None:
    """None when this server can accept image inputs, else the client-facing reason."""
    if "image" in config.served_modalities:
        return None
    if config.mm.text_model_only:
        return "image input is disabled on this server (--text-model-only)"
    if "vision" in config.mm.disabled_encoders:
        return "image input is disabled on this server (--mm-disable)"
    return "the served model has no vision support"


def _check_media_domain(url: str, config: ServerArgs) -> None:
    """Reject a URL whose hostname is not in --allowed-media-domains; empty allowlist admits any domain."""
    from urllib.parse import urlparse

    raw = config.allowed_media_domains
    allowed = {d.strip().lower().rstrip(".") for d in raw.split(",") if d.strip()}
    if not allowed:
        return
    host = (urlparse(url).hostname or "").rstrip(".")
    if host not in allowed:
        raise ValueError(
            f"the URL must be from one of the allowed domains: {sorted(allowed)}; "
            f"input URL domain: {host or '<none>'}"
        )


def _check_redirect_target(url: str, config: ServerArgs) -> None:
    """A redirect hop is the remote server's choice, not the client's URL: it must pass the
    allowlist like the original, and may never land on a loopback / private / link-local
    address (an allowed public host 302-ing into the operator's network is the SSRF shape)."""
    import ipaddress
    from urllib.parse import urlparse

    if not url.startswith(("http://", "https://")):
        raise ValueError("the URL redirects to a non-http(s) location")
    try:
        _check_media_domain(url, config)
    except ValueError:
        raise ValueError("the URL redirects outside the allowed domains") from None
    host = (urlparse(url).hostname or "").rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("the URL redirects to a private or loopback address")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # a name; the allowlist above is the gate for names (no DNS resolution here)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast:  # covers loopback, private, link-local, CGNAT, unspecified
        raise ValueError("the URL redirects to a private or loopback address")


async def _fetch_remote_media(url: str, config: ServerArgs) -> bytes:
    """GET an http(s) ref, following redirects one hop at a time so every target is re-checked."""
    import httpx

    _check_media_domain(url, config)
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            async with client.stream("GET", url) as resp:
                if resp.next_request is not None:  # a 3xx with a Location: httpx built the hop
                    url = str(resp.next_request.url)
                    _check_redirect_target(url, config)
                    continue
                resp.raise_for_status()
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > _MAX_IMAGE_BYTES:
                    raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > _MAX_IMAGE_BYTES:
                        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
                return bytes(buf)
    raise ValueError(f"the URL redirects more than {_MAX_REDIRECTS} times")


def _load_local_media(url: str, config: ServerArgs) -> bytes:
    """Read a file:// ref; the resolved path must be a strict subpath of --allowed-local-media-path."""
    from pathlib import Path
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    root = config.allowed_local_media_path
    if not root:
        raise ValueError("cannot load local files without --allowed-local-media-path")
    spec = urlparse(url)
    filepath = Path(url2pathname((spec.netloc or "") + (spec.path or "")))
    # resolve() follows symlinks, so a link inside the root escaping it is rejected too
    resolved = filepath.resolve()
    if Path(root).resolve() not in resolved.parents:
        raise ValueError("the file path must be a subpath of --allowed-local-media-path")
    return resolved.read_bytes()


async def _fetch_one(ref: dict[str, Any], config: ServerArgs) -> bytes:
    """One ref to bytes. Our own ValueErrors are already client-safe and pass through; anything
    a library raises is reduced to its failure class and logged with the detail."""
    data = ref.get("data") or ""
    try:
        if ref.get("kind") == "b64":
            return base64.b64decode(data)
        if data.startswith("data:"):
            return base64.b64decode(data.split(",", 1)[1])
    except Exception as exc:  # noqa: BLE001 -- binascii.Error / IndexError, input-driven
        raise ValueError("image data is not valid base64") from exc
    if data.startswith(("http://", "https://")):
        import httpx

        try:
            return await _fetch_remote_media(data, config)
        except ValueError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"the image URL returned HTTP {exc.response.status_code}") from exc
        except httpx.TimeoutException as exc:
            raise ValueError("timed out fetching the image URL") from exc
        except Exception as exc:  # noqa: BLE001 -- transport / protocol / bad URL
            logger.warning("image URL fetch failed: %r", exc)
            raise ValueError("the image URL is unreachable") from exc
    if data.startswith("file://"):
        try:
            return _load_local_media(data, config)
        except ValueError:
            raise
        except OSError as exc:
            logger.warning("local image read failed: %r", exc)
            raise ValueError("the local file is missing or unreadable") from exc
    raise ValueError("unsupported image source (expect http(s)/file url or base64)")


async def fetch_image_bytes(refs: list[dict[str, Any]], config: ServerArgs) -> list[bytes]:
    """Resolve collected image refs (URLs / base64) to raw bytes, in order."""
    out: list[bytes] = []
    for ref in refs:
        try:
            out.append(await _fetch_one(ref, config))
        except ValueError as exc:
            raise ValueError(f"could not load image: {exc}") from exc
    return out


__all__ = ["collect_image_refs", "fetch_image_bytes", "image_reject_reason"]
