from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import re
import secrets
import signal
import time
import unicodedata
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import uvicorn
import websockets
from websockets.protocol import State as WsState
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from websockets.exceptions import ConnectionClosed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_TIMEOUTS: dict[str, int] = {
    # GPT-5 family
    "gpt-5.4-pro": 2400,
    "gpt-5-4-pro": 2400,
    "gpt-5.3": 600,
    "gpt-5-3": 600,
    "gpt-5.2": 300,
    "gpt-5-2": 300,
    "gpt-5.1": 300,
    "gpt-5-1": 300,
    "gpt-5": 300,
    "gpt-5-mini": 120,
    # GPT-4.5 family
    "gpt-4.5-pro": 600,
    "gpt-4.5": 300,
    # GPT-4.1 family
    "gpt-4.1": 180,
    "gpt-4.1-mini": 120,
    "gpt-4.1-nano": 60,
    # GPT-4o family
    "gpt-4o": 180,
    "gpt-4o-mini": 120,
    # o1 family
    "o1": 600,
    "o1-pro": 1200,
    "o1-mini": 300,
    "o1-preview": 600,
    # o3 family
    "o3": 600,
    "o3-pro": 1200,
    "o3-mini": 300,
    # o4 family
    "o4-mini": 300,
    # auto
    "auto": 300,
}

_DEFAULT_MODELS = (
    "gpt-5.4-pro",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.5-pro",
    "gpt-4o",
    "o3",
    "o3-mini",
    "o4-mini",
    "auto",
)

# Models known to trigger deep-research / web-browsing mode.
# These need longer stable-poll windows and browsing-phase detection.
_DEEP_RESEARCH_MODELS: frozenset[str] = frozenset({
    "gpt-5.4-pro",
    "gpt-5-4-pro",
    "gpt-5.3",
    "gpt-5-3",
    "gpt-4.5-pro",
    "o1-pro",
    "o3-pro",
})


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    auth_key: str
    host: str
    api_port: int
    allowed_origins: tuple[str, ...]
    browser_http_host: str
    browser_port_override: int | None
    devtools_port_file: str
    browser_connect_timeout: float
    browser_call_timeout: float
    browser_ws_max_size: int
    browser_ws_ping_interval: float
    browser_ws_ping_timeout: float
    browser_ws_close_timeout: float
    request_cooldown_seconds: float
    max_queue_size: int
    page_ready_timeout_seconds: float
    page_create_timeout_seconds: float
    default_timeout_seconds: int
    models_cache_ttl_seconds: float
    default_models: tuple[str, ...]
    model_timeouts: dict[str, int]
    rate_limit_cooldown_seconds: float
    prompt_chunk_size: int
    max_prompt_chars: int
    log_level: str
    debug: bool
    chatgpt_origin: str


class SettingsError(RuntimeError):
    pass


class ProxyError(RuntimeError):
    status_code = 500
    error_type = "proxy_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequestError(ProxyError):
    status_code = 400
    error_type = "invalid_request_error"


class AuthenticationError(ProxyError):
    status_code = 401
    error_type = "authentication_error"


class QueueFullError(ProxyError):
    status_code = 503
    error_type = "queue_full"


class BrowserUnavailableError(ProxyError):
    status_code = 503
    error_type = "browser_unavailable"


class PageStateError(ProxyError):
    status_code = 502
    error_type = "page_state_error"


class RateLimitedError(ProxyError):
    status_code = 429
    error_type = "rate_limit_error"

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class PromptTooLargeError(ProxyError):
    status_code = 413
    error_type = "prompt_too_large"


class JobTimeoutError(ProxyError):
    status_code = 504
    error_type = "timeout"


class JobCancelledError(ProxyError):
    status_code = 499
    error_type = "cancelled"


class CdpDisconnectedError(ProxyError):
    status_code = 502
    error_type = "cdp_disconnected"


class CdpTimeoutError(ProxyError):
    status_code = 504
    error_type = "cdp_timeout"


class CdpCallError(ProxyError):
    status_code = 502
    error_type = "cdp_call_error"

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def load_settings() -> Settings:
    auth_key = os.getenv("CDP_AUTH_KEY", "").strip()
    if not auth_key:
        raise SettingsError(
            "CDP_AUTH_KEY is required. Refusing to start without an explicit API key."
        )

    if auth_key == "sk-chatgpt-pro":
        raise SettingsError(
            "CDP_AUTH_KEY must not use the insecure default value 'sk-chatgpt-pro'."
        )

    host = os.getenv("CDP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    api_port = int(os.getenv("CDP_API_PORT", "5006"))
    browser_http_host = os.getenv("CDP_BROWSER_HTTP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    browser_port_override_raw = os.getenv("CDP_BROWSER_PORT", "").strip()
    browser_port_override = int(browser_port_override_raw) if browser_port_override_raw else None
    devtools_port_file = os.path.expanduser(
        os.getenv(
            "CDP_DEVTOOLS_PORT_FILE",
            "~/Library/Application Support/adspower_global/cwd_global/"
            "source/cache/k116el91_h1n96yr/DevToolsActivePort",
        )
    )

    default_timeout_seconds = int(os.getenv("CDP_DEFAULT_TIMEOUT", "300"))
    model_timeouts = dict(_DEFAULT_MODEL_TIMEOUTS)
    model_timeouts_override = os.getenv("CDP_MODEL_TIMEOUTS_JSON", "").strip()
    if model_timeouts_override:
        try:
            model_timeouts.update(
                {str(k): int(v) for k, v in json.loads(model_timeouts_override).items()}
            )
        except Exception as exc:  # pragma: no cover - defensive only
            raise SettingsError(f"Invalid CDP_MODEL_TIMEOUTS_JSON: {exc}") from exc

    settings = Settings(
        auth_key=auth_key,
        host=host,
        api_port=api_port,
        allowed_origins=_env_csv("CDP_CORS_ORIGINS", ""),
        browser_http_host=browser_http_host,
        browser_port_override=browser_port_override,
        devtools_port_file=devtools_port_file,
        browser_connect_timeout=float(os.getenv("CDP_BROWSER_CONNECT_TIMEOUT", "15")),
        browser_call_timeout=float(os.getenv("CDP_BROWSER_CALL_TIMEOUT", "20")),
        browser_ws_max_size=int(os.getenv("CDP_BROWSER_WS_MAX_SIZE", str(2**25))),
        browser_ws_ping_interval=float(os.getenv("CDP_BROWSER_WS_PING_INTERVAL", "30")),
        browser_ws_ping_timeout=float(os.getenv("CDP_BROWSER_WS_PING_TIMEOUT", "120")),
        browser_ws_close_timeout=float(os.getenv("CDP_BROWSER_WS_CLOSE_TIMEOUT", "10")),
        request_cooldown_seconds=float(os.getenv("CDP_COOLDOWN", "15")),
        max_queue_size=int(os.getenv("CDP_MAX_QUEUE", "16")),
        page_ready_timeout_seconds=float(os.getenv("CDP_PAGE_READY_TIMEOUT", "60")),
        page_create_timeout_seconds=float(os.getenv("CDP_PAGE_CREATE_TIMEOUT", "20")),
        default_timeout_seconds=default_timeout_seconds,
        models_cache_ttl_seconds=float(os.getenv("CDP_MODELS_CACHE_TTL", "300")),
        default_models=_env_csv("CDP_DEFAULT_MODELS", ",".join(_DEFAULT_MODELS)) or _DEFAULT_MODELS,
        model_timeouts=model_timeouts,
        rate_limit_cooldown_seconds=float(os.getenv("CDP_RATE_LIMIT_COOLDOWN", "120")),
        prompt_chunk_size=int(os.getenv("CDP_PROMPT_CHUNK_SIZE", "4000")),
        max_prompt_chars=int(os.getenv("CDP_MAX_PROMPT_CHARS", "200000")),
        log_level=os.getenv("CDP_LOG_LEVEL", "INFO").upper(),
        debug=_env_bool("CDP_DEBUG", False),
        chatgpt_origin=os.getenv("CDP_CHATGPT_ORIGIN", "https://chatgpt.com/").rstrip("/") + "/",
    )
    return settings


SETTINGS = load_settings()

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cdp_proxy")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class _Metrics:
    """Simple in-memory request metrics (thread-safe via asyncio single-thread)."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.total_latency: float = 0.0
        self.model_requests: dict[str, int] = {}
        self.model_errors: dict[str, int] = {}
        self._start_time: float = time.monotonic()

    def record_request(self, model: str, latency: float, error: bool = False) -> None:
        self.total_requests += 1
        self.total_latency += latency
        self.model_requests[model] = self.model_requests.get(model, 0) + 1
        if error:
            self.total_errors += 1
            self.model_errors[model] = self.model_errors.get(model, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        avg_latency = (self.total_latency / self.total_requests) if self.total_requests else 0.0
        error_rate = (self.total_errors / self.total_requests) if self.total_requests else 0.0
        return {
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(error_rate, 4),
            "avg_latency_seconds": round(avg_latency, 3),
            "by_model": {
                model: {
                    "requests": count,
                    "errors": self.model_errors.get(model, 0),
                }
                for model, count in self.model_requests.items()
            },
        }


METRICS = _Metrics()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    return time.monotonic()


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


# Regex patterns for post-processing assistant text
# Matches "Thought for N seconds/minutes" prefix lines produced by thinking models
_RE_THOUGHT_PREFIX = re.compile(
    r'^Thought for [\d.]+ (?:second|minute)s?\s*\n+',
    re.IGNORECASE,
)
# Matches full lines that are browsing/searching status (produced during deep-research phase)
_RE_BROWSING_LINE = re.compile(
    r'^(?:Browsing|Searching|Reading|Analyzing|Gathering|Looking up|Fetching|Visiting)[^\n]{0,200}\n?',
    re.IGNORECASE | re.MULTILINE,
)
# Matches inline citation badge patterns produced by ChatGPT deep-research:
#   single-line:  "Bank for International Settlements+2"  "arXiv+3"
#   multi-line:   "Federal Reserve\n+2\nStripe\n+3\n"  (innerText of citation pill <a>)
_RE_CITATION_INLINE = re.compile(
    r'(?:[\w][^\n+]{0,80}\+\d+)+'          # single-line: SourceName+N[SourceName+N...]
    r'|(?:^[^\n+\d]{1,80}\n\+\d+\n?)+',    # multi-line:  SourceName\n+N\n
    re.MULTILINE,
)
# Matches standalone "+N" lines that are leftover superscript numbers
_RE_PLUS_NUMBER_LINE = re.compile(r'^\+\d+\s*$', re.MULTILINE)
# Matches 【N†sourceName】 style citations
_RE_CITATION_BRACKET = re.compile(r'【\d+†[^】]{0,100}】')


def _is_only_citations(text: str) -> bool:
    """Return True when text consists entirely of citation-badge noise (no real content).

    ChatGPT deep-research citation pills render as innerText like:
        "Federal Reserve\\n+2\\nStripe\\n+3\\n..."
    After stripping those patterns, if less than 40 printable chars remain
    the response is pure browsing-phase noise.
    """
    cleaned = _RE_CITATION_INLINE.sub('', text)
    cleaned = _RE_PLUS_NUMBER_LINE.sub('', cleaned)
    cleaned = _RE_CITATION_BRACKET.sub('', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return len(cleaned) < 40


def _clean_assistant_text(text: str) -> str:
    """Remove deep-research noise: thinking prefix, browsing status lines, citation badges."""
    if not text:
        return text
    # 1. Strip "Thought for N seconds" prefix
    text = _RE_THOUGHT_PREFIX.sub('', text)
    # 2. Remove browsing status lines
    text = _RE_BROWSING_LINE.sub('', text)
    # 3. Remove inline citation badges (single-line and multi-line variants)
    text = _RE_CITATION_INLINE.sub('', text)
    # 4. Remove leftover standalone "+N" lines
    text = _RE_PLUS_NUMBER_LINE.sub('', text)
    # 5. Remove bracket citations 【N†source】
    text = _RE_CITATION_BRACKET.sub('', text)
    # 6. Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 7. Strip trailing whitespace on each line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()


def _sanitize_excerpt(value: str, limit: int = 160) -> str:
    cleaned = _normalize_text(value).replace("\n", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _masked_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _parse_timeout_seconds(payload: dict[str, Any], model: str) -> int:
    override = payload.get("proxy_timeout_seconds")
    if override is not None:
        try:
            parsed = int(override)
        except (TypeError, ValueError):
            raise BadRequestError("proxy_timeout_seconds must be an integer")
        if parsed <= 0:
            raise BadRequestError("proxy_timeout_seconds must be > 0")
        return parsed
    return SETTINGS.model_timeouts.get(model, SETTINGS.default_timeout_seconds)


# Maps API-facing model names to the internal ChatGPT cookie slug.
# Only entries that differ need to be listed; unlisted names are passed through.
_MODEL_SLUG_MAP: dict[str, str] = {
    # GPT-5 family (dot→dash normalisation used by some callers)
    "gpt-5-4-pro": "gpt-5.4-pro",
    "gpt-5-3": "gpt-5.3",
    "gpt-5-2": "gpt-5.2",
    "gpt-5-1": "gpt-5.1",
    # GPT-4.1 family — ChatGPT UI may use dotted form
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    # o-series aliases
    "o1-preview": "o1-preview",
    # Common external aliases
    "chatgpt-4o-latest": "gpt-4o",
    "gpt-4-turbo": "gpt-4o",
}


def _model_cookie_payload(model_name: str) -> str:
    effort = "extended"
    base = model_name
    if base.endswith("-low"):
        effort = "low"
        base = base.removesuffix("-low")
    elif base.endswith("-default"):
        effort = "default"
        base = base.removesuffix("-default")
    slug = _MODEL_SLUG_MAP.get(base, base)
    return urllib.parse.quote(json.dumps({"model": slug, "effort": effort}, separators=(",", ":")))


def _poll_interval(timeout_seconds: int, elapsed_seconds: float) -> float:
    if timeout_seconds >= 600:
        if elapsed_seconds < 30:
            return 1.0
        if elapsed_seconds < 120:
            return 2.5
        if elapsed_seconds < 600:
            return 5.0
        return 10.0
    if elapsed_seconds < 15:
        return 0.75
    return 1.5


_RATE_LIMIT_PATTERNS = (
    "too many requests",
    "please wait",
    "slow down",
    "rate limit",
    "you’re making requests",
    "limit reached",
    "usage cap",
    "you’ve reached your limit",
    "you have reached your limit",
    "daily limit",
    "message limit",
    "reached the limit",
)


def _looks_like_rate_limit(text: str) -> bool:
    lowered = _normalize_text(text).lower()
    return any(pattern in lowered for pattern in _RATE_LIMIT_PATTERNS)


def _sse_payload(data: str) -> str:
    return f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# Request content shaping
# ---------------------------------------------------------------------------


def _flatten_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            part_type = str(item.get("type", "")).strip().lower()
            if part_type in {"text", "input_text"}:
                parts.append(str(item.get("text") or item.get("content") or ""))
            elif part_type in {"image_url", "input_image"}:
                image_payload = item.get("image_url")
                if isinstance(image_payload, dict):
                    url = image_payload.get("url") or ""
                else:
                    url = image_payload or item.get("url") or ""
                if url:
                    parts.append(f"[Image omitted by proxy: {url}]")
                else:
                    parts.append("[Image omitted by proxy]")
            elif part_type in {"refusal", "audio"}:
                parts.append(f"[{part_type} content omitted by proxy]")
            else:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
                else:
                    parts.append(f"[Unsupported content part: {part_type or 'unknown'}]")
        return "\n".join(part for part in parts if part)
    return str(content)


def build_prompt_from_messages(messages: list[dict[str, Any]] | Any) -> str:
    if not isinstance(messages, list):
        raise BadRequestError("messages must be an array")
    if not messages:
        raise BadRequestError("messages is required")

    system_instructions: list[str] = []
    transcript: list[tuple[str, str]] = []

    for raw_msg in messages:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role", "user")).strip().lower() or "user"
        content = _flatten_message_content(raw_msg.get("content"))
        if role in {"system", "developer"}:
            if content.strip():
                label = "Developer" if role == "developer" else "System"
                system_instructions.append(f"{label} instructions:\n{content.strip()}")
            continue
        if role == "tool":
            name = str(raw_msg.get("name") or "tool").strip()
            if content.strip():
                transcript.append((f"Tool ({name})", content.strip()))
            continue
        label = "Assistant" if role == "assistant" else "User"
        transcript.append((label, content.strip()))

    if not transcript:
        raise BadRequestError("No user-visible messages were provided")

    last_role, last_text = transcript[-1]
    if last_role != "User":
        raise BadRequestError("The final message must be from the user")

    if not system_instructions and len(transcript) == 1:
        prompt = last_text
    else:
        sections: list[str] = []
        if system_instructions:
            sections.append("\n\n".join(system_instructions))
        sections.append("Conversation transcript:")
        for role, text in transcript:
            sections.append(f"{role}:\n{text}")
        sections.append("Respond as the assistant to the last user message above.")
        prompt = "\n\n".join(section for section in sections if section.strip())

    prompt = _normalize_text(prompt).strip()
    if not prompt:
        raise BadRequestError("No usable user prompt was found")
    if len(prompt) > SETTINGS.max_prompt_chars:
        raise PromptTooLargeError(
            f"Prompt is too large ({len(prompt)} chars > {SETTINGS.max_prompt_chars})"
        )
    return prompt


# ---------------------------------------------------------------------------
# CDP client
# ---------------------------------------------------------------------------


def _scan_chrome_devtools_port_sync(host: str, timeout: float = 1.5) -> int | None:
    """Auto-discover a running Chrome DevTools endpoint via OS listening ports.

    Uses ``lsof`` (macOS/Linux) to enumerate TCP-LISTEN ports, then probes
    only those for a valid ``/json/version`` response.  Falls back to a small
    set of well-known ports if ``lsof`` is unavailable.  Completes in < 5 s.
    """
    import subprocess

    listening_ports: list[int] = []

    # --- Strategy 1: OS-level enumeration (fast, no network I/O) ----------
    try:
        proc = subprocess.run(
            ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fn"],
            capture_output=True, text=True, timeout=5,
        )
        for line in proc.stdout.splitlines():
            # lsof -Fn emits lines like "n*:9222" or "n127.0.0.1:20050"
            if line.startswith("n") and ":" in line:
                port_str = line.rsplit(":", 1)[-1]
                try:
                    listening_ports.append(int(port_str))
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # lsof not available — try ss (Linux)
        try:
            proc = subprocess.run(
                ["ss", "-tlnH"],
                capture_output=True, text=True, timeout=5,
            )
            for line in proc.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    addr = parts[3]  # e.g. "127.0.0.1:9222" or "*:9222"
                    port_str = addr.rsplit(":", 1)[-1]
                    try:
                        listening_ports.append(int(port_str))
                    except ValueError:
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # De-duplicate and sort; prioritise common CDP / AdsPower ranges
    priority_ports = list(range(9222, 9232)) + list(range(20000, 20010)) + list(range(50000, 50010))
    if listening_ports:
        # Put priority ports first, then the rest of OS-discovered ports
        priority_set = set(priority_ports)
        ordered: list[int] = []
        seen: set[int] = set()
        for p in priority_ports:
            if p in listening_ports and p not in seen:
                ordered.append(p)
                seen.add(p)
        for p in sorted(set(listening_ports)):
            if p not in seen:
                ordered.append(p)
                seen.add(p)
        candidate_ports = ordered
    else:
        # Fallback: only probe well-known ports (no full scan)
        candidate_ports = priority_ports

    for port in candidate_ports:
        try:
            url = f"http://{host}:{port}/json/version"
            payload = _http_get_json_sync(url, timeout)
            ws_url = str(payload.get("webSocketDebuggerUrl") or "")
            if ws_url:
                logger.info("Auto-discovered Chrome DevTools on port %s", port)
                return port
        except Exception:
            pass
    return None


def _read_cdp_port_sync(settings: Settings) -> int:
    if settings.browser_port_override is not None:
        return settings.browser_port_override
    # 1. Try the DevToolsActivePort file (standard AdsPower location)
    try:
        with open(settings.devtools_port_file, "r", encoding="utf-8") as handle:
            port = int(handle.readline().strip())
            # Verify the port is actually reachable before returning it
            try:
                _http_get_json_sync(
                    f"http://{settings.browser_http_host}:{port}/json/version",
                    timeout=2.0,
                )
                return port
            except Exception:
                logger.warning(
                    "DevToolsActivePort file says %s but port is unreachable — scanning...", port
                )
    except FileNotFoundError:
        logger.warning("DevToolsActivePort file not found (%s) — auto-scanning ports...", settings.devtools_port_file)
    except ValueError as exc:
        raise BrowserUnavailableError(
            f"DevToolsActivePort file is malformed: {settings.devtools_port_file}"
        ) from exc
    # 2. Fallback: auto-scan to find a Chrome DevTools endpoint
    discovered = _scan_chrome_devtools_port_sync(settings.browser_http_host)
    if discovered is not None:
        return discovered
    raise BrowserUnavailableError(
        f"Cannot find Chrome DevTools. "
        f"Make sure AdsPower browser is open. "
        f"(DevToolsActivePort file: {settings.devtools_port_file})"
    )


def _http_get_json_sync(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class CdpBrowserClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._ws: websockets.ClientConnection | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._counter = itertools.count(1)
        self._connect_lock = asyncio.Lock()
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._ws.state is WsState.OPEN and self._reader_task is not None and not self._reader_task.done()

    async def _discover_browser_ws_url(self) -> str:
        port = await asyncio.to_thread(_read_cdp_port_sync, self.settings)
        url = f"http://{self.settings.browser_http_host}:{port}/json/version"
        try:
            payload = await asyncio.to_thread(
                _http_get_json_sync, url, self.settings.browser_connect_timeout
            )
        except urllib.error.URLError as exc:
            raise BrowserUnavailableError(
                f"Cannot reach Chrome DevTools endpoint at {url}: {exc}"
            ) from exc
        ws_url = str(payload.get("webSocketDebuggerUrl") or "").strip()
        if not ws_url:
            raise BrowserUnavailableError("DevTools /json/version did not return browser websocket URL")
        return ws_url

    async def connect(self) -> None:
        if self.connected:
            return
        async with self._connect_lock:
            if self.connected:
                return
            await self.close()
            ws_url = await self._discover_browser_ws_url()
            try:
                self._ws = await websockets.connect(
                    ws_url,
                    max_size=self.settings.browser_ws_max_size,
                    ping_interval=self.settings.browser_ws_ping_interval,
                    ping_timeout=self.settings.browser_ws_ping_timeout,
                    close_timeout=self.settings.browser_ws_close_timeout,
                    open_timeout=self.settings.browser_connect_timeout,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                raise BrowserUnavailableError(f"Failed to connect to browser websocket: {exc}") from exc
            self._reader_task = asyncio.create_task(self._reader_loop(), name="cdp-browser-reader")
            self._closed = False
            await self.call("Browser.getVersion", timeout=10)
            logger.info("Connected to browser CDP websocket")

    async def reconnect(self) -> None:
        async with self._connect_lock:
            await self.close()
            self._ws = None
            self._reader_task = None
        await self.connect()

    async def close(self) -> None:
        self._closed = True
        reader = self._reader_task
        ws = self._ws
        self._reader_task = None
        self._ws = None
        await self._fail_pending(CdpDisconnectedError("CDP connection closed"))
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
        if ws is not None and ws.state is not WsState.CLOSED:
            with contextlib.suppress(Exception):
                await ws.close()

    async def _fail_pending(self, exc: Exception) -> None:
        pending = list(self._pending.items())
        self._pending.clear()
        for _, future in pending:
            if not future.done():
                future.set_exception(exc)

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        ws = self._ws
        try:
            async for raw in ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Ignoring malformed CDP message")
                    continue
                msg_id = message.get("id")
                if msg_id is None:
                    continue
                future = self._pending.pop(int(msg_id), None)
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message.get("error") or {}
                    future.set_exception(
                        CdpCallError(
                            str(error.get("message") or "CDP call failed"),
                            code=error.get("code"),
                        )
                    )
                else:
                    future.set_result(message)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:
            if not self._closed:
                logger.warning("Browser websocket closed: %s", exc)
                await self._fail_pending(CdpDisconnectedError(f"Browser websocket closed: {exc}"))
        except Exception as exc:  # pragma: no cover - network dependent
            if not self._closed:
                logger.exception("Browser websocket reader failed")
                await self._fail_pending(CdpDisconnectedError(f"Browser websocket reader failed: {exc}"))
        finally:
            if not self._closed:
                await self._fail_pending(CdpDisconnectedError("Browser websocket reader stopped"))

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        await self.connect()
        if self._ws is None:
            raise CdpDisconnectedError("Browser websocket is not connected")
        deadline = timeout if timeout is not None else self.settings.browser_call_timeout
        msg_id = next(self._counter)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future
        payload: dict[str, Any] = {"id": msg_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id
        try:
            await self._ws.send(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        except Exception as exc:
            self._pending.pop(msg_id, None)
            raise CdpDisconnectedError(f"Failed to send CDP command {method}: {exc}") from exc
        try:
            return await asyncio.wait_for(future, timeout=deadline)
        except asyncio.TimeoutError as exc:
            self._pending.pop(msg_id, None)
            raise CdpTimeoutError(f"Timed out waiting for CDP response to {method}") from exc


# ---------------------------------------------------------------------------
# Page session helpers
# ---------------------------------------------------------------------------



_BOOTSTRAP_TEMPLATE = r"""
(() => {
  if (!window.__OAI_PROXY) {
    window.__OAI_PROXY = (() => {
      const normalize = (value) => String(value ?? '')
        .replace(/\u00a0/g, ' ')
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n');
      const tidy = (value) => normalize(value)
        .replace(/[ \t]+\n/g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
      const visible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
      const queryFirst = (selectors, root=document) => {
        for (const sel of selectors) {
          const el = root.querySelector(sel);
          if (visible(el)) return el;
        }
        return null;
      };
      const composerSelectors = [
        '#prompt-textarea',
        '[data-testid="prompt-textarea"]',
        'div#prompt-textarea[contenteditable="true"]',
        'div[contenteditable="true"][data-testid="prompt-textarea"]',
        'div[contenteditable="true"][aria-label*="Message"]',
        'textarea[data-testid="prompt-textarea"]',
        'textarea[placeholder*="Message"]',
      ];
      const turnSelector = '[data-testid*="conversation-turn"], [data-message-author-role], [data-turn]';
      const pruneSelectors = [
        'button',
        'nav',
        'aside',
        'form',
        '[role="button"]',
        '[data-testid*="toolbar"]',
        '[data-testid*="copy"]',
        '[data-testid*="thumb"]',
        '[data-testid*="retry"]',
        '[aria-label*="Copy"]',
        '[aria-label*="Like"]',
        '[aria-label*="Dislike"]',
        '[aria-label*="More"]',
        '[aria-label*="Read aloud"]',
        '[aria-label*="Edit"]',
        // Citation pills produced by deep-research / web-browsing mode
        '[data-testid="webpage-citation-pill"]',
        '[data-testid*="citation"]',
        '[data-testid*="source-pill"]',
        // Superscript citation numbers and footnote markers
        'sup',
        '[class*="footnote"]',
        '[class*="citation"]',
        // Browsing / searching status indicators
        '[role="status"]',
        '[aria-live="polite"]',
        '[data-testid*="browsing"]',
        '[data-testid*="thinking-indicator"]',
        '[data-testid*="progress"]',
      ].join(',');
      const findComposer = () => queryFirst(composerSelectors);
      const textOf = (el) => {
        if (!el) return '';
        if ('value' in el && !el.isContentEditable) return tidy(el.value || '');
        return tidy(el.innerText || el.textContent || '');
      };
      const stripLeadingLabels = (value) => tidy(value)
        .replace(/^(ChatGPT said:|You said:)\s*/i, '')
        .trim();
      const roleOf = (node) => {
        if (!node) return '';
        const direct = node.getAttribute('data-message-author-role') || node.getAttribute('data-turn');
        if (direct) return String(direct).toLowerCase();
        const nested = node.querySelector('[data-message-author-role],[data-turn]');
        if (nested) return String(nested.getAttribute('data-message-author-role') || nested.getAttribute('data-turn') || '').toLowerCase();
        const testid = String(node.getAttribute('data-testid') || '');
        if (/assistant/i.test(testid)) return 'assistant';
        if (/user/i.test(testid)) return 'user';
        return '';
      };
      const stripNode = (node) => {
        const clone = node.cloneNode(true);
        clone.querySelectorAll(pruneSelectors).forEach((el) => el.remove());
        return clone;
      };
      const extractTurnText = (node) => {
        if (!node) return '';
        const clone = stripNode(node);
        // Targeted content selectors — prefer these over the full node text
        const targetedSelectors = '.markdown, [class*="markdown"], [data-testid="message-content"], article, pre, code, table, figure, figcaption, [data-testid*="artifact"]';
        const targetedCandidates = [];
        const allCandidates = [];
        const pushAll = (el) => {
          const text = stripLeadingLabels(textOf(el));
          if (text) allCandidates.push(text);
        };
        const pushTargeted = (el) => {
          const text = stripLeadingLabels(textOf(el));
          if (text) { targetedCandidates.push(text); allCandidates.push(text); }
        };
        pushAll(clone);
        clone.querySelectorAll(targetedSelectors).forEach(pushTargeted);
        const fallbackArtifactLabels = [];
        clone.querySelectorAll('canvas, svg, img').forEach((el) => {
          const label = el.getAttribute('aria-label') || el.getAttribute('alt') || el.getAttribute('title') || '';
          if (label.trim()) fallbackArtifactLabels.push(label.trim());
        });
        if (!allCandidates.length && fallbackArtifactLabels.length) allCandidates.push(fallbackArtifactLabels.join('\n'));
        if (!allCandidates.length) return '';
        // Prefer the longest targeted candidate (e.g. .markdown content) when it
        // is substantial — avoids picking the full-clone text that may include
        // citation-pill noise when deep-research is active.
        if (targetedCandidates.length) {
          targetedCandidates.sort((a, b) => b.length - a.length);
          const best = targetedCandidates[0];
          if (best.length >= 50) return best;
        }
        allCandidates.sort((a, b) => b.length - a.length);
        return allCandidates[0];
      };
      // Check whether the last assistant turn's .markdown contains real prose
      // (paragraphs, headings, lists, tables) vs. just citation links.
      const hasSubstantialContent = (node) => {
        if (!node) return false;
        const md = node.querySelector('.markdown, [class*="markdown"]');
        if (!md) return false;
        const clone = md.cloneNode(true);
        // Remove citation pills and superscripts before checking
        clone.querySelectorAll(pruneSelectors).forEach((el) => el.remove());
        // Check for structural HTML elements that indicate real written content
        const substantialTags = 'p, h1, h2, h3, h4, h5, h6, li, td, th, dd, dt, blockquote, pre, div';
        const elements = clone.querySelectorAll(substantialTags);
        let realTextLen = 0;
        for (const el of elements) {
          const t = (el.innerText || el.textContent || '').trim();
          if (t.length > 5) realTextLen += t.length;
        }
        return realTextLen >= 40;
      };
      const turnRoots = () => {
        const raw = Array.from(document.querySelectorAll(turnSelector));
        const seen = new Set();
        const result = [];
        for (const node of raw) {
          if (!node || seen.has(node)) continue;
          const parent = node.parentElement ? node.parentElement.closest(turnSelector) : null;
          if (parent) continue;
          const role = roleOf(node);
          if (!role || (role !== 'user' && role !== 'assistant' && role !== 'tool')) continue;
          seen.add(node);
          result.push(node);
        }
        return result;
      };
      const dismissStartupDialogs = () => {
        const dialogs = Array.from(document.querySelectorAll('[role="dialog"]'));
        for (const dialog of dialogs) {
          const text = tidy(dialog.innerText || '');
          if (!text) continue;
          const isRateLimit = /too many requests|please wait|slow down|rate limit|limit reached|usage cap|daily limit/i.test(text);
          // Always try to click dismiss buttons — even rate-limit dialogs must be
          // acknowledged so the UI becomes interactive again.
          const buttons = Array.from(dialog.querySelectorAll('button'));
          let clickedLabel = '';
          for (const button of buttons) {
            const label = tidy(button.innerText || button.getAttribute('aria-label') || '').toLowerCase();
            if (['got it', 'dismiss', 'close', 'not now', 'ok', 'okay', 'continue', 'i understand'].includes(label)
                || /got.?it|okay|dismiss/i.test(label)) {
              button.click();
              clickedLabel = label;
              break;
            }
          }
          if (isRateLimit) {
            return { dismissed: !!clickedLabel, blocked: true, text, button: clickedLabel };
          }
          if (clickedLabel) {
            return { dismissed: true, blocked: false, text, button: clickedLabel };
          }
        }
        return { dismissed: false, blocked: false, text: '' };
      };
      const clearComposer = () => {
        const el = findComposer();
        if (!el) return { ok: false, reason: 'composer-not-found' };
        el.focus();
        if (el.isContentEditable) {
          const range = document.createRange();
          range.selectNodeContents(el);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          try { document.execCommand('delete', false); } catch (_) {}
          el.innerHTML = '';
          el.textContent = '';
          el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }));
        } else if ('value' in el) {
          el.value = '';
          el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
        return { ok: true, text: textOf(el) };
      };
      const focusComposer = () => {
        const el = findComposer();
        if (!el) return { ok: false, reason: 'composer-not-found' };
        el.focus();
        if (el.isContentEditable) {
          const range = document.createRange();
          range.selectNodeContents(el);
          range.collapse(false);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
        }
        return { ok: true, text: textOf(el) };
      };
      const setComposerTextDirect = (text) => {
        const el = findComposer();
        if (!el) return { ok: false, reason: 'composer-not-found' };
        if (el.isContentEditable) {
          el.innerHTML = '';
          el.textContent = text;
          el.dispatchEvent(new InputEvent('input', { bubbles: true, data: text, inputType: 'insertText' }));
        } else if ('value' in el) {
          el.value = text;
          el.dispatchEvent(new InputEvent('input', { bubbles: true, data: text, inputType: 'insertText' }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
        return { ok: true, text: textOf(el) };
      };
      const setComposerTextProseMirror = (text) => {
        const el = findComposer();
        if (!el) return { ok: false, reason: 'composer-not-found' };
        el.focus();
        if (el.isContentEditable) {
          while (el.firstChild) { el.removeChild(el.firstChild); }
          const lines = text.split('\n');
          for (const line of lines) {
            const p = document.createElement('p');
            p.textContent = line || '\u200B';
            el.appendChild(p);
          }
          el.dispatchEvent(new InputEvent('input', {
            bubbles: true, cancelable: true,
            inputType: 'insertText', data: text,
          }));
          const range = document.createRange();
          range.selectNodeContents(el);
          range.collapse(false);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
        } else if ('value' in el) {
          el.value = text;
          el.dispatchEvent(new InputEvent('input', { bubbles: true, data: text, inputType: 'insertText' }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
        return { ok: true, text: textOf(el) };
      };
      const clickSendButton = () => {
        const sendSelectors = [
          'button[data-testid="send-button"]',
          'button[aria-label="Send prompt"]',
          'button[aria-label="Send"]',
          'button.send-button',
          'form button[type="submit"]',
        ];
        for (const sel of sendSelectors) {
          const btn = document.querySelector(sel);
          if (btn && visible(btn) && !btn.disabled) {
            btn.click();
            return { clicked: true, via: sel };
          }
        }
        const buttons = Array.from(document.querySelectorAll('button'));
        for (const btn of buttons) {
          if (!visible(btn) || btn.disabled) continue;
          const label = (btn.getAttribute('aria-label') || '').toLowerCase();
          const testId = btn.getAttribute('data-testid') || '';
          if (label.includes('send') || testId.includes('send')) {
            btn.click();
            return { clicked: true, via: label || testId };
          }
        }
        return { clicked: false };
      };
      const clickNewChat = () => {
        const candidates = Array.from(document.querySelectorAll('button, a'));
        for (const el of candidates) {
          const label = tidy(el.innerText || el.getAttribute('aria-label') || '');
          const testid = String(el.getAttribute('data-testid') || '');
          const href = String(el.getAttribute('href') || '');
          if (/new chat/i.test(label) || /new-chat/i.test(testid) || label === 'New chat') {
            el.click();
            return { clicked: true, via: label || testid || href };
          }
        }
        return { clicked: false };
      };
      const snapshot = () => {
        const roots = turnRoots();
        const bodyText = tidy(document.body ? document.body.innerText || '' : '').slice(0, 1200);
        const turns = roots.map((node) => ({ role: roleOf(node), text: extractTurnText(node) }));
        const userTurns = turns.filter((t) => t.role === 'user');
        const assistantTurns = turns.filter((t) => t.role === 'assistant');
        const lastAssistantRoot = assistantTurns.length ? roots.filter((n) => roleOf(n) === 'assistant').pop() : null;
        const dialog = document.querySelector('[role="dialog"]');
        const dialogText = dialog ? tidy(dialog.innerText || '') : '';
        const composer = findComposer();
        return {
          url: String(window.location.href || ''),
          title: String(document.title || ''),
          readyState: String(document.readyState || ''),
          bodySample: bodyText,
          composerPresent: !!composer,
          composerText: textOf(composer),
          turnCount: turns.length,
          userTurnCount: userTurns.length,
          assistantTurnCount: assistantTurns.length,
          lastUserText: userTurns.length ? userTurns[userTurns.length - 1].text : '',
          lastAssistantText: assistantTurns.length ? assistantTurns[assistantTurns.length - 1].text : '',
          hasSubstantialContent: hasSubstantialContent(lastAssistantRoot),
          streaming: !!document.querySelector('[data-testid="stop-button"], button[aria-label*="Stop"]'),
          artifactCount: roots.length ? (roots[roots.length - 1]?.querySelectorAll('canvas, svg, [data-testid*="artifact"]').length || 0) : 0,
          dialogText,
          blockedByRateLimit: /too many requests|please wait|slow down|rate limit|limit/i.test(dialogText),
        };
      };
      return {
        normalize,
        tidy,
        findComposer,
        textOf,
        dismissStartupDialogs,
        clearComposer,
        focusComposer,
        setComposerTextDirect,
        setComposerTextProseMirror,
        clickSendButton,
        clickNewChat,
        snapshot,
      };
    })();
  }
  return __BODY__;
})()
"""


def _bootstrap_js(body: str) -> str:
    return _BOOTSTRAP_TEMPLATE.replace("__BODY__", body)


@dataclass()
class PageSession:
    browser: CdpBrowserClient
    target_id: str
    session_id: str
    origin: str

    @classmethod
    async def create(cls, browser: CdpBrowserClient, origin: str, timeout: float) -> "PageSession":
        create_resp = await browser.call(
            "Target.createTarget",
            {"url": origin},
            timeout=timeout,
        )
        target_id = str(create_resp["result"]["targetId"])
        attach_resp = await browser.call(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
            timeout=timeout,
        )
        session_id = str(attach_resp["result"]["sessionId"])
        page = cls(browser=browser, target_id=target_id, session_id=session_id, origin=origin)
        await page._enable_domains()
        await browser.call("Target.activateTarget", {"targetId": target_id}, timeout=timeout)
        await page.call("Page.bringToFront", timeout=timeout)
        return page

    @staticmethod
    def _is_recoverable_session_error(exc: Exception) -> bool:
        if isinstance(exc, (CdpDisconnectedError, CdpTimeoutError)):
            return True
        if isinstance(exc, CdpCallError):
            lowered = exc.message.lower()
            return (
                "session" in lowered
                or "target" in lowered
                or "context" in lowered
                or "closed" in lowered
            )
        return False

    async def _enable_domains(self) -> None:
        await asyncio.gather(
            self.browser.call("Page.enable", session_id=self.session_id, timeout=10),
            self.browser.call("Runtime.enable", session_id=self.session_id, timeout=10),
            self.browser.call("DOM.enable", session_id=self.session_id, timeout=10),
            self.browser.call(
                "Network.enable",
                {"maxTotalBufferSize": 0, "maxResourceBufferSize": 0},
                session_id=self.session_id,
                timeout=10,
            ),
        )

    async def reattach(self) -> None:
        await self.browser.reconnect()
        attach_resp = await self.browser.call(
            "Target.attachToTarget",
            {"targetId": self.target_id, "flatten": True},
            timeout=10,
        )
        self.session_id = str(attach_resp["result"]["sessionId"])
        await self._enable_domains()
        with contextlib.suppress(ProxyError):
            await self.browser.call("Target.activateTarget", {"targetId": self.target_id}, timeout=5)
        with contextlib.suppress(ProxyError):
            await self.call("Page.bringToFront", timeout=5)

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await self.browser.call(
                    method,
                    params,
                    session_id=self.session_id,
                    timeout=timeout,
                )
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and self._is_recoverable_session_error(exc):
                    logger.warning(
                        "Recovering page session after %s on %s",
                        exc.__class__.__name__,
                        method,
                    )
                    await self.reattach()
                    continue
                raise
        assert last_exc is not None
        raise PageStateError(f"Page call failed for {method}: {last_exc}")

    async def evaluate(
        self,
        expression: str,
        *,
        await_promise: bool = False,
        timeout_ms: int = 30000,
    ) -> Any:
        response = await self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
                "timeout": timeout_ms,
                "userGesture": True,
            },
            timeout=max(timeout_ms / 1000 + 10, 10),
        )
        runtime_result = response.get("result", {})
        exception_details = runtime_result.get("exceptionDetails")
        if exception_details:
            text = exception_details.get("text") or runtime_result.get("result", {}).get("description") or "JavaScript evaluation failed"
            raise CdpCallError(str(text))
        result = runtime_result.get("result", {})
        if result.get("subtype") == "error":
            raise CdpCallError(str(result.get("description") or "JavaScript evaluation returned error"))
        if "value" in result:
            return result["value"]
        return None

    async def snapshot(self) -> dict[str, Any]:
        value = await self.evaluate(_bootstrap_js("window.__OAI_PROXY.snapshot()"), await_promise=False)
        if not isinstance(value, dict):
            raise PageStateError("Page snapshot did not return an object")
        return value

    async def dismiss_startup_dialogs(self) -> dict[str, Any]:
        value = await self.evaluate(
            _bootstrap_js("window.__OAI_PROXY.dismissStartupDialogs()"),
            await_promise=False,
        )
        return value if isinstance(value, dict) else {"dismissed": False}

    async def clear_composer(self) -> dict[str, Any]:
        value = await self.evaluate(
            _bootstrap_js("window.__OAI_PROXY.clearComposer()"),
            await_promise=False,
        )
        return value if isinstance(value, dict) else {"ok": False}

    async def focus_composer(self) -> dict[str, Any]:
        value = await self.evaluate(
            _bootstrap_js("window.__OAI_PROXY.focusComposer()"),
            await_promise=False,
        )
        return value if isinstance(value, dict) else {"ok": False}

    async def click_new_chat(self) -> dict[str, Any]:
        value = await self.evaluate(
            _bootstrap_js("window.__OAI_PROXY.clickNewChat()"),
            await_promise=False,
        )
        return value if isinstance(value, dict) else {"clicked": False}

    async def set_composer_text_direct(self, text: str) -> dict[str, Any]:
        payload = json.dumps(text, ensure_ascii=False)
        value = await self.evaluate(
            _bootstrap_js(f"window.__OAI_PROXY.setComposerTextDirect({payload})"),
            await_promise=False,
        )
        return value if isinstance(value, dict) else {"ok": False}

    async def set_composer_text_prosemirror(self, text: str) -> dict[str, Any]:
        payload = json.dumps(text, ensure_ascii=False)
        value = await self.evaluate(
            _bootstrap_js(f"window.__OAI_PROXY.setComposerTextProseMirror({payload})"),
            await_promise=False,
        )
        return value if isinstance(value, dict) else {"ok": False}

    async def click_send_button(self) -> dict[str, Any]:
        value = await self.evaluate(
            _bootstrap_js("window.__OAI_PROXY.clickSendButton()"),
            await_promise=False,
        )
        return value if isinstance(value, dict) else {"clicked": False}

    async def navigate(self, url: str) -> None:
        await self.call("Page.navigate", {"url": url}, timeout=20)

    async def insert_text(self, text: str, chunk_size: int) -> None:
        for idx in range(0, len(text), chunk_size):
            chunk = text[idx : idx + chunk_size]
            await self.call("Input.insertText", {"text": chunk}, timeout=20)

    async def press_enter(self) -> None:
        await self.call(
            "Input.dispatchKeyEvent",
            {
                "type": "keyDown",
                "key": "Enter",
                "code": "Enter",
                "windowsVirtualKeyCode": 13,
                "nativeVirtualKeyCode": 13,
            },
            timeout=10,
        )
        await self.call(
            "Input.dispatchKeyEvent",
            {"type": "keyUp", "key": "Enter", "code": "Enter"},
            timeout=10,
        )

    async def close(self) -> None:
        with contextlib.suppress(ProxyError, Exception):
            await self.browser.call("Target.closeTarget", {"targetId": self.target_id}, timeout=10)


# ---------------------------------------------------------------------------
# Job queue and worker
# ---------------------------------------------------------------------------


@dataclass()
class StreamEvent:
    kind: Literal["delta", "done", "error"]
    data: str = ""
    finish_reason: str | None = None


@dataclass()
class ChatJob:
    id: str
    model: str
    prompt: str
    timeout_seconds: int
    stream: bool
    future: asyncio.Future[str]
    cancel_event: asyncio.Event
    created_at: float = field(default_factory=_now)
    progress_queue: asyncio.Queue[StreamEvent] | None = None


class BrowserWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.browser = CdpBrowserClient(settings)
        self.queue: asyncio.Queue[ChatJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self._runner_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._last_prompt_sent_at = 0.0
        self.active_job: ChatJob | None = None
        self._rate_limited_until: float = 0.0
        self._consecutive_rate_limits: int = 0
        self._persistent_page: PageSession | None = None
        self._last_model: str | None = None
        self._models_cache: tuple[float, list[str]] | None = None
        self._models_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.create_task(self._run(), name="cdp-browser-worker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self.active_job is not None:
            self.active_job.cancel_event.set()
            if self.active_job.progress_queue is not None:
                await self.active_job.progress_queue.put(
                    StreamEvent(kind="error", data="Proxy is shutting down")
                )
            if not self.active_job.future.done():
                self.active_job.future.set_exception(JobCancelledError("Proxy is shutting down"))
        if self._runner_task is not None:
            self._runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner_task
        while not self.queue.empty():
            try:
                job = self.queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - defensive only
                break
            if not job.future.done():
                job.future.set_exception(JobCancelledError("Proxy is shutting down"))
            self.queue.task_done()
        if self._persistent_page is not None:
            with contextlib.suppress(Exception):
                await self._persistent_page.close()
            self._persistent_page = None
        await self.browser.close()

    async def enqueue_chat(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
        stream: bool,
    ) -> ChatJob:
        if self.queue.full():
            raise QueueFullError("Proxy queue is full")
        loop = asyncio.get_running_loop()
        job = ChatJob(
            id=f"job_{uuid.uuid4().hex[:12]}",
            model=model,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            stream=stream,
            future=loop.create_future(),
            cancel_event=asyncio.Event(),
            progress_queue=asyncio.Queue() if stream else None,
        )
        self.queue.put_nowait(job)
        logger.info(
            "Enqueued chat job %s model=%s queue_depth=%s",
            job.id,
            model,
            self.queue.qsize(),
        )
        return job

    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                job = await self.queue.get()
                self.active_job = job
                try:
                    if job.cancel_event.is_set():
                        raise JobCancelledError("Request cancelled before execution started")
                    result = await self._process_chat_job(job)
                    if not job.future.done():
                        job.future.set_result(result)
                    if job.progress_queue is not None:
                        await job.progress_queue.put(StreamEvent(kind="done", finish_reason="stop"))
                except Exception as exc:
                    if job.progress_queue is not None:
                        await job.progress_queue.put(StreamEvent(kind="error", data=str(exc)))
                    if not job.future.done():
                        job.future.set_exception(exc)
                finally:
                    self.active_job = None
                    self.queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _wait_for_cooldown(self, cancel_event: asyncio.Event) -> None:
        remaining = self.settings.request_cooldown_seconds - (_now() - self._last_prompt_sent_at)
        while remaining > 0:
            if cancel_event.is_set():
                raise JobCancelledError("Request cancelled during cooldown")
            await asyncio.sleep(min(0.5, remaining))
            remaining = self.settings.request_cooldown_seconds - (_now() - self._last_prompt_sent_at)

    async def _open_chat_page(self, model: str | None = None) -> PageSession:
        await self.browser.connect()
        page = await PageSession.create(
            self.browser,
            origin="about:blank",
            timeout=self.settings.page_create_timeout_seconds,
        )
        if model:
            await self._set_model_cookie(page, model)
        await page.navigate(self.settings.chatgpt_origin)
        return page

    async def _find_existing_chatgpt_page(self) -> PageSession | None:
        try:
            result = await self.browser.call("Target.getTargets", timeout=10)
            targets = result.get("result", {}).get("targetInfos", [])
            for target in targets:
                url = str(target.get("url") or "")
                if target.get("type") == "page" and "chatgpt.com" in url:
                    target_id = target["targetId"]
                    attach_resp = await self.browser.call(
                        "Target.attachToTarget",
                        {"targetId": target_id, "flatten": True},
                        timeout=10,
                    )
                    session_id = str(attach_resp["result"]["sessionId"])
                    page = PageSession(
                        browser=self.browser,
                        target_id=target_id,
                        session_id=session_id,
                        origin=self.settings.chatgpt_origin,
                    )
                    await page._enable_domains()
                    logger.info("Reusing existing chatgpt.com tab: %s", target_id)
                    return page
        except Exception as exc:
            logger.debug("Could not find existing chatgpt page: %s", exc)
        return None

    async def _acquire_page(self, model: str) -> PageSession:
        await self.browser.connect()

        # 1. Try reusing persistent page
        if self._persistent_page is not None:
            try:
                snapshot = await self._persistent_page.snapshot()
                if snapshot.get("composerPresent") or snapshot.get("readyState") == "complete":
                    if model != self._last_model:
                        await self._set_model_cookie(self._persistent_page, model)
                        await self._persistent_page.click_new_chat()
                        await asyncio.sleep(0.5)
                        self._last_model = model
                    logger.info("Reusing persistent page for model=%s", model)
                    return self._persistent_page
            except Exception as exc:
                logger.warning("Persistent page unusable: %s", exc)
                with contextlib.suppress(Exception):
                    await self._persistent_page.close()
                self._persistent_page = None

        # 2. Try finding existing chatgpt.com tab in browser
        page = await self._find_existing_chatgpt_page()
        if page is not None:
            try:
                snapshot = await page.snapshot()
                if snapshot.get("composerPresent"):
                    await self._set_model_cookie(page, model)
                    self._persistent_page = page
                    self._last_model = model
                    return page
            except Exception:
                with contextlib.suppress(Exception):
                    await page.close()

        # 3. Create new page as last resort
        page = await self._open_chat_page(model=model)
        self._persistent_page = page
        self._last_model = model
        return page

    async def _set_model_cookie(self, page: PageSession, model: str) -> None:
        logger.debug("Setting model cookie: model=%s deep_research=%s", model, model in _DEEP_RESEARCH_MODELS)
        payload = _model_cookie_payload(model)
        response = await page.call(
            "Network.setCookie",
            {
                "name": "oai-last-model-config",
                "value": payload,
                "url": self.settings.chatgpt_origin,
                "domain": ".chatgpt.com",
                "path": "/",
                "secure": True,
                "sameSite": "Lax",
            },
            timeout=10,
        )
        success = bool(response.get("result", {}).get("success"))
        if not success:
            raise PageStateError(f"Failed to set model cookie for {model}")

    async def _wait_for_page_ready(self, page: PageSession, cancel_event: asyncio.Event) -> dict[str, Any]:
        deadline = _now() + self.settings.page_ready_timeout_seconds
        last_snapshot: dict[str, Any] | None = None
        while _now() < deadline:
            if cancel_event.is_set():
                raise JobCancelledError("Request cancelled while waiting for page")
            snapshot = await page.snapshot()
            last_snapshot = snapshot
            if snapshot.get("blockedByRateLimit"):
                raise RateLimitedError(snapshot.get("dialogText") or "ChatGPT rate limited the session")
            body_sample = str(snapshot.get("bodySample") or "")
            lower_body = body_sample.lower()
            current_url = str(snapshot.get("url") or "")
            if any(x in current_url for x in ("/auth/login", "/auth/signin")) or "log in" in lower_body:
                raise PageStateError("ChatGPT session is not logged in")
            if "verify you are human" in lower_body or "just a moment" in lower_body:
                raise PageStateError("Cloudflare / anti-bot challenge is blocking the page")
            if _looks_like_rate_limit(body_sample):
                raise RateLimitedError(body_sample[:200])
            if snapshot.get("composerPresent"):
                await page.dismiss_startup_dialogs()
                return snapshot
            await asyncio.sleep(0.5)
        context = last_snapshot or {}
        raise PageStateError(
            "Timed out waiting for ChatGPT composer "
            f"(url={context.get('url')!r}, title={context.get('title')!r})"
        )

    async def _ensure_fresh_chat(self, page: PageSession, cancel_event: asyncio.Event) -> dict[str, Any]:
        snapshot = await self._wait_for_page_ready(page, cancel_event)
        if snapshot.get("turnCount", 0):
            clicked = await page.click_new_chat()
            if clicked.get("clicked"):
                await asyncio.sleep(1)
                snapshot = await self._wait_for_page_ready(page, cancel_event)
        if snapshot.get("turnCount", 0):
            await page.navigate(self.settings.chatgpt_origin)
            await asyncio.sleep(1)
            snapshot = await self._wait_for_page_ready(page, cancel_event)
        if snapshot.get("turnCount", 0):
            raise PageStateError("Could not obtain a fresh empty chat page")
        return snapshot

    async def _type_prompt(self, page: PageSession, prompt: str) -> None:
        expected = _normalize_text(prompt)
        for attempt in range(3):
            # Strategy 1: ProseMirror DOM injection (best for contentEditable)
            pm_result = await page.set_composer_text_prosemirror(prompt)
            if pm_result.get("ok"):
                snapshot = await page.snapshot()
                actual = _normalize_text(str(snapshot.get("composerText") or ""))
                if actual == expected:
                    return
                logger.warning(
                    "ProseMirror typing mismatch on attempt %s: expected=%s actual=%s",
                    attempt + 1, len(expected), len(actual),
                )

            # Strategy 2: Direct textContent set
            fallback = await page.set_composer_text_direct(prompt)
            if fallback.get("ok"):
                actual_direct = _normalize_text(str(fallback.get("text") or ""))
                if actual_direct == expected:
                    return

            # Strategy 3: CDP Input.insertText (legacy fallback)
            clear_result = await page.clear_composer()
            if clear_result.get("ok"):
                focus_result = await page.focus_composer()
                if focus_result.get("ok"):
                    await page.insert_text(prompt, self.settings.prompt_chunk_size)
                    snapshot = await page.snapshot()
                    actual = _normalize_text(str(snapshot.get("composerText") or ""))
                    if actual == expected:
                        return

            logger.warning("All typing strategies failed on attempt %s", attempt + 1)
            await asyncio.sleep(0.5)
        raise PageStateError("Failed to set prompt text after 3 attempts (ProseMirror + Direct + CDP)")

    async def _wait_for_user_turn(
        self,
        page: PageSession,
        *,
        prompt: str,
        previous_user_turn_count: int,
        cancel_event: asyncio.Event,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        expected = _normalize_text(prompt).strip()
        deadline = _now() + timeout_seconds
        while _now() < deadline:
            if cancel_event.is_set():
                raise JobCancelledError("Request cancelled while waiting for send confirmation")
            snapshot = await page.snapshot()
            if snapshot.get("blockedByRateLimit"):
                raise RateLimitedError(snapshot.get("dialogText") or "ChatGPT rate limited the session")
            last_user = _normalize_text(str(snapshot.get("lastUserText") or "")).strip()
            user_turns = int(snapshot.get("userTurnCount") or 0)
            if user_turns > previous_user_turn_count and last_user == expected:
                return snapshot
            await asyncio.sleep(0.25)
        raise PageStateError("Prompt send could not be confirmed from the page")

    @staticmethod
    def _compute_stream_delta(previous_emitted: str, current_text: str) -> tuple[str, str]:
        if current_text.startswith(previous_emitted):
            return current_text[len(previous_emitted) :], current_text
        max_prefix = min(len(previous_emitted), len(current_text))
        common = 0
        for idx in range(max_prefix):
            if previous_emitted[idx] != current_text[idx]:
                break
            common = idx + 1
        return current_text[common:], current_text

    async def _collect_assistant_response(
        self,
        page: PageSession,
        *,
        model: str = "",
        timeout_seconds: int,
        cancel_event: asyncio.Event,
        stream_queue: asyncio.Queue[StreamEvent] | None,
    ) -> str:
        start = _now()
        deadline = start + timeout_seconds
        last_text = ""
        emitted_text = ""
        stable_polls = 0
        empty_done_polls = 0

        is_deep_research = model in _DEEP_RESEARCH_MODELS or timeout_seconds >= 1200
        # Deep-research models need more stable polls; standard models just 2
        if is_deep_research:
            required_stable_polls = 5
        elif timeout_seconds >= 600:
            required_stable_polls = 3
        else:
            required_stable_polls = 2
        # Deep-research: wait at least this many seconds before counting stable polls
        min_wait_seconds = 30 if is_deep_research else 0
        max_empty_polls = 40 if is_deep_research else (20 if timeout_seconds >= 600 else 5)

        while _now() < deadline:
            if cancel_event.is_set():
                raise JobCancelledError("Request cancelled while awaiting assistant response")
            elapsed = _now() - start
            await asyncio.sleep(_poll_interval(timeout_seconds, elapsed))
            snapshot = await page.snapshot()
            dialog_text = str(snapshot.get("dialogText") or "")
            body_sample = str(snapshot.get("bodySample") or "")
            if (dialog_text and _looks_like_rate_limit(dialog_text)) or _looks_like_rate_limit(body_sample):
                raise RateLimitedError(dialog_text or body_sample[:200])
            if "something went wrong" in body_sample.lower():
                raise PageStateError("ChatGPT returned 'Something went wrong'")
            text = _normalize_text(str(snapshot.get("lastAssistantText") or ""))

            # Deep-research browsing-phase guard:
            # Two complementary heuristics — Python regex on text AND JS DOM
            # structural check via hasSubstantialContent.  Both must agree the
            # content is real before we count it as stable progress.
            is_citation_noise = bool(text) and _is_only_citations(text)
            has_substantial = bool(snapshot.get("hasSubstantialContent"))
            still_browsing = is_citation_noise or (is_deep_research and bool(text) and not has_substantial)
            if still_browsing:
                logger.debug(
                    "Browsing phase detected (citation_noise=%s, substantial=%s, %d chars) — resetting stable_polls",
                    is_citation_noise, has_substantial, len(text),
                )
                # Don't update last_text; reset stability counter; continue waiting
                stable_polls = 0
                streaming = bool(snapshot.get("streaming"))
                if not streaming and elapsed < max(min_wait_seconds, 10):
                    pass  # keep waiting even if stop-button gone
                continue

            if text != last_text:
                last_text = text
                stable_polls = 0
                empty_done_polls = 0
                if stream_queue is not None and text:
                    delta, emitted_text = self._compute_stream_delta(emitted_text, text)
                    if delta:
                        await stream_queue.put(StreamEvent(kind="delta", data=delta))
            else:
                stable_polls += 1

            streaming = bool(snapshot.get("streaming"))
            if streaming:
                continue

            # Don't start counting stability until minimum wait elapsed (deep-research)
            if elapsed < min_wait_seconds:
                continue

            if text:
                if stable_polls >= required_stable_polls:
                    return _clean_assistant_text(text)
            else:
                empty_done_polls += 1
                if snapshot.get("artifactCount") and empty_done_polls >= 2:
                    return "[The assistant returned a non-text artifact/canvas response with no extractable DOM text. Open the ChatGPT page to inspect it.]"
                if empty_done_polls >= max_empty_polls:
                    break

        if last_text:
            return _clean_assistant_text(last_text)
        raise JobTimeoutError(
            f"Timed out waiting for assistant response "
            f"(model={model!r}, timeout={timeout_seconds}s, "
            f"last_text_chars={len(last_text)}, stable_polls={stable_polls})"
        )

    async def _cancel_page_generation(self, page: PageSession) -> None:
        with contextlib.suppress(Exception):
            await page.evaluate(
                _bootstrap_js(
                    "(() => { const btn = document.querySelector('[data-testid=\"stop-button\"], button[aria-label*=\"Stop\"]'); if (btn) { btn.click(); return true; } return false; })()"
                ),
                await_promise=False,
            )

    async def _process_chat_job(self, job: ChatJob) -> str:
        remaining = self._rate_limited_until - _now()
        if remaining > 0:
            raise RateLimitedError(
                f"ChatGPT rate limit cooldown active, retry after {remaining:.0f}s",
                retry_after=int(remaining) + 1,
            )

        send_confirmed = False
        for attempt in range(2):
            if job.cancel_event.is_set():
                raise JobCancelledError("Request cancelled before browser interaction")
            page: PageSession | None = None
            created_new_page = False
            try:
                await self._wait_for_cooldown(job.cancel_event)
                page = await self._acquire_page(job.model)
                created_new_page = (page is not self._persistent_page)
                await self._ensure_fresh_chat(page, job.cancel_event)
                await self._type_prompt(page, job.prompt)
                before_send = await page.snapshot()
                previous_user_turn_count = int(before_send.get("userTurnCount") or 0)
                send_result = await page.click_send_button()
                if not send_result.get("clicked"):
                    logger.warning("Send button not found, falling back to Enter key")
                    await page.press_enter()
                self._last_prompt_sent_at = _now()
                confirmed = await self._wait_for_user_turn(
                    page,
                    prompt=job.prompt,
                    previous_user_turn_count=previous_user_turn_count,
                    cancel_event=job.cancel_event,
                )
                send_confirmed = True
                logger.info(
                    "Job %s prompt sent model=%s assistant_turns=%s",
                    job.id,
                    job.model,
                    confirmed.get("assistantTurnCount"),
                )
                result = await self._collect_assistant_response(
                    page,
                    model=job.model,
                    timeout_seconds=job.timeout_seconds,
                    cancel_event=job.cancel_event,
                    stream_queue=job.progress_queue,
                )
                self._consecutive_rate_limits = 0
                logger.info(
                    "Job %s completed model=%s response_chars=%d elapsed=%.1fs",
                    job.id, job.model, len(result), _now() - job.created_at,
                )
                # Keep page alive for reuse (don't close)
                self._persistent_page = page
                return result
            except JobCancelledError:
                if page is not None:
                    await self._cancel_page_generation(page)
                raise
            except RateLimitedError as exc:
                self._consecutive_rate_limits += 1
                multiplier = min(2 ** (self._consecutive_rate_limits - 1), 5)
                cooldown = self.settings.rate_limit_cooldown_seconds * multiplier
                self._rate_limited_until = _now() + cooldown
                exc.retry_after = int(cooldown) + 1
                logger.warning(
                    "Rate limited on job %s (consecutive=%s), cooldown for %ss",
                    job.id,
                    self._consecutive_rate_limits,
                    cooldown,
                )
                # Don't close page on rate limit — it's still usable later
                raise
            except (CdpDisconnectedError, CdpTimeoutError, PageStateError) as exc:
                recoverable = not send_confirmed and attempt == 0
                if recoverable:
                    logger.warning(
                        "Retrying job %s before send confirmation after %s",
                        job.id,
                        exc,
                    )
                    # Invalidate persistent page on session errors
                    self._persistent_page = None
                    if page is not None:
                        await page.close()
                    await asyncio.sleep(1)
                    continue
                # Invalidate persistent page on unrecoverable errors
                self._persistent_page = None
                if page is not None:
                    await page.close()
                raise
        raise PageStateError("Browser interaction failed before prompt send")

    async def get_models(self) -> list[str]:
        cached = self._models_cache
        if cached is not None and (_now() - cached[0]) < self.settings.models_cache_ttl_seconds:
            return list(cached[1])
        async with self._models_lock:
            cached = self._models_cache
            if cached is not None and (_now() - cached[0]) < self.settings.models_cache_ttl_seconds:
                return list(cached[1])
            try:
                models = await self._fetch_models_live()
            except Exception as exc:
                logger.warning("Failed to fetch models dynamically: %s", exc)
                if cached is not None:
                    return list(cached[1])
                return list(self.settings.default_models)
            self._models_cache = (_now(), models)
            return list(models)

    async def _fetch_models_live(self) -> list[str]:
        page = await self._open_chat_page()
        try:
            await self._wait_for_page_ready(page, asyncio.Event())
            result = await page.evaluate(
                """
(async () => {
  const response = await fetch('/backend-api/models?history_and_training_disabled=true', {
    credentials: 'include',
    headers: { 'accept': 'application/json' },
  });
  if (!response.ok) {
    return { ok: false, status: response.status, body: await response.text().catch(() => '') };
  }
  const payload = await response.json();
  return { ok: true, models: Array.isArray(payload.models) ? payload.models.map((m) => m.slug).filter(Boolean) : [] };
})()
""",
                await_promise=True,
                timeout_ms=30000,
            )
            if not isinstance(result, dict) or not result.get("ok"):
                raise PageStateError(f"Model fetch failed: {result}")
            live_models = [str(model).strip() for model in result.get("models", []) if str(model).strip()]
            if not live_models:
                raise PageStateError("ChatGPT returned an empty model list")
            # Merge with default models to ensure full coverage — live models
            # take priority (appear first), then any defaults not already present.
            seen: set[str] = set(live_models)
            merged = list(live_models)
            for m in self.settings.default_models:
                if m not in seen:
                    merged.append(m)
                    seen.add(m)
            return merged
        finally:
            await page.close()

    def health_snapshot(self) -> dict[str, Any]:
        active_age = None
        if self.active_job is not None:
            active_age = round(_now() - self.active_job.created_at, 3)
        rl_remaining = self._rate_limited_until - _now()
        return {
            "browser_connected": self.browser.connected,
            "queue_depth": self.queue.qsize(),
            "max_queue_size": self.settings.max_queue_size,
            "active_job": self.active_job.id if self.active_job else None,
            "active_job_age_seconds": active_age,
            "cooldown_seconds": self.settings.request_cooldown_seconds,
            "rate_limit_cooldown_remaining": round(max(rl_remaining, 0), 1),
            "consecutive_rate_limits": self._consecutive_rate_limits,
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


async def require_auth(request: Request) -> None:
    auth_header = request.headers.get("authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthenticationError("Missing Bearer token")
    if not secrets.compare_digest(token.strip(), SETTINGS.auth_key):
        raise AuthenticationError("Invalid API key")


async def _watch_disconnect(request: Request, cancel_event: asyncio.Event) -> None:
    try:
        while not cancel_event.is_set():
            if await request.is_disconnected():
                cancel_event.set()
                return
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        raise


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    worker = BrowserWorker(SETTINGS)
    await worker.start()
    app.state.worker = worker
    logger.info(
        "CDP proxy starting host=%s port=%s cors=%s auth=%s",
        SETTINGS.host,
        SETTINGS.api_port,
        SETTINGS.allowed_origins or (),
        _masked_secret(SETTINGS.auth_key),
    )
    # Graceful shutdown on SIGTERM (Docker/K8s sends this before SIGKILL)
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler(sig: int) -> None:
        logger.info("Received signal %s — initiating graceful shutdown", signal.Signals(sig).name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler, sig)

    try:
        yield
    finally:
        logger.info("Shutting down worker…")
        await worker.stop()
        logger.info("CDP proxy stopped")


app = FastAPI(lifespan=lifespan)

if SETTINGS.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.allowed_origins),
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
    )


from starlette.middleware.base import BaseHTTPMiddleware


class _RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach an X-Request-Id header to every response (echo or generate)."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


app.add_middleware(_RequestIdMiddleware)


def _get_worker(request: Request) -> BrowserWorker:
    worker = getattr(request.app.state, "worker", None)
    if worker is None:
        raise BrowserUnavailableError("Worker is not initialized")
    return worker


@app.exception_handler(ProxyError)
async def proxy_error_handler(_: Request, exc: ProxyError):
    payload = {"error": {"message": exc.message, "type": exc.error_type}}
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitedError) and exc.retry_after is not None:
        headers["Retry-After"] = str(exc.retry_after)
    return JSONResponse(status_code=exc.status_code, content=payload, headers=headers or None)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {"error": {"message": str(exc.detail), "type": "http_error"}}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.get("/health")
async def health(request: Request):
    worker = _get_worker(request)
    return {
        "status": "ok",
        **worker.health_snapshot(),
    }


@app.get("/metrics")
async def metrics(request: Request):
    worker = _get_worker(request)
    return {
        **METRICS.snapshot(),
        **worker.health_snapshot(),
    }


@app.get("/v1/models")
async def list_models(request: Request, _: None = Depends(require_auth)):
    worker = _get_worker(request)
    models = await worker.get_models()
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "openai"}
            for model in models
        ],
    }


def _make_completion(model: str, content: str) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _make_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _: None = Depends(require_auth)):
    worker = _get_worker(request)
    try:
        payload = await request.json()
    except Exception as exc:
        raise BadRequestError(f"Request body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BadRequestError("JSON body must be an object")

    model = str(payload.get("model") or "auto").strip() or "auto"
    stream = bool(payload.get("stream", False))
    prompt = build_prompt_from_messages(payload.get("messages") or [])
    timeout_seconds = _parse_timeout_seconds(payload, model)

    job = await worker.enqueue_chat(
        model=model,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        stream=stream,
    )

    logger.info(
        "Accepted request job=%s model=%s stream=%s timeout=%ss prompt_chars=%s",
        job.id,
        model,
        stream,
        timeout_seconds,
        len(prompt),
    )

    disconnect_task = asyncio.create_task(_watch_disconnect(request, job.cancel_event))

    request_id = getattr(request.state, "request_id", "")

    if not stream:
        request_start = time.monotonic()
        is_error = False
        try:
            result = await job.future
            return JSONResponse(content=_make_completion(model, result))
        except Exception:
            is_error = True
            raise
        finally:
            METRICS.record_request(model, time.monotonic() - request_start, error=is_error)
            disconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnect_task

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def event_stream():
        stream_start = time.monotonic()
        is_error = False
        try:
            yield _sse_payload(json.dumps(_make_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={"role": "assistant"},
                finish_reason=None,
            ), ensure_ascii=False))
            assert job.progress_queue is not None
            while True:
                event = await job.progress_queue.get()
                if event.kind == "delta":
                    if event.data:
                        chunk = _make_chunk(
                            completion_id=completion_id,
                            created=created,
                            model=model,
                            delta={"content": event.data},
                            finish_reason=None,
                        )
                        yield _sse_payload(json.dumps(chunk, ensure_ascii=False))
                    continue
                if event.kind == "done":
                    chunk = _make_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=model,
                        delta={},
                        finish_reason=event.finish_reason or "stop",
                    )
                    yield _sse_payload(json.dumps(chunk, ensure_ascii=False))
                    yield _sse_payload("[DONE]")
                    return
                if event.kind == "error":
                    is_error = True
                    error_payload = {
                        "error": {
                            "message": event.data,
                            "type": "stream_error",
                        }
                    }
                    yield _sse_payload(json.dumps(error_payload, ensure_ascii=False))
                    yield _sse_payload("[DONE]")
                    return
        except asyncio.CancelledError:
            is_error = True
            job.cancel_event.set()
            raise
        finally:
            METRICS.record_request(model, time.monotonic() - stream_start, error=is_error)
            disconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnect_task
            if not job.future.done():
                job.cancel_event.set()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    logger.info("Starting CDP proxy on http://%s:%s", SETTINGS.host, SETTINGS.api_port)
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.api_port, log_level=SETTINGS.log_level.lower())
