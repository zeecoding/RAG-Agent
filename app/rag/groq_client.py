"""Thin async client for Groq's OpenAI-compatible chat completions API.

Groq is a *hosted* inference API (you call their endpoint; they run the
model on their hardware) — not Ollama. No local or remote server to manage,
which is why this replaces the earlier Ollama client for a no-GPU setup.

Lifecycle follows the same pattern as db/pool.py:
  - init_groq()       → call from FastAPI lifespan startup
  - get_groq_client() → accessor, raises if not initialized
  - close_groq()      → call from FastAPI lifespan shutdown

Key-rotation strategy (multi-key):
  - Keys are rotated round-robin across chat()/chat_structured() calls.
  - On a 429, the offending key enters a per-key cooldown (keyed on
    time.monotonic()) and the *next* available key is tried immediately.
  - If every key is simultaneously cooling down the call falls back to the
    existing exponential-backoff sleep against the key with the soonest
    expiry, then retries once the earliest key is available again.
  - Non-429 errors (5xx, network errors) keep the existing backoff-and-
    retry behaviour on the *same* key — they are not rate-limit signals.
"""
import asyncio
import json
import time
import httpx
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_client: "GroqClient | None" = None


def _prepare_schema_for_groq(schema: dict) -> dict:
    """Sanitize a Pydantic-generated JSON schema for Groq strict mode.

    Groq's strict structured outputs require:
      - No 'title' fields (Pydantic adds them at top-level and per-property)
      - 'additionalProperties': false on all object types
      - All properties listed in 'required'
      - No '$defs' / '$ref' (inline everything)
    """
    import copy
    schema = copy.deepcopy(schema)

    def _clean(node: dict) -> dict:
        node.pop("title", None)
        node.pop("default", None)
        if node.get("type") == "object":
            node["additionalProperties"] = False
            for prop in node.get("properties", {}).values():
                _clean(prop)
        if "items" in node and isinstance(node["items"], dict):
            _clean(node["items"])
        return node

    # Resolve $defs if present (Pydantic v2 can generate these for nested models)
    defs = schema.pop("$defs", None)
    schema = _clean(schema)
    return schema


def _extract_retry_delay(response: httpx.Response, fallback_delay: float) -> float:
    """Extract retry delay from headers or response text if available."""
    retry_header = response.headers.get("retry-after")
    if retry_header:
        try:
            return max(float(retry_header), 1.0)
        except ValueError:
            pass
    # Groq error messages often state: "Please try again in 4.23s"
    try:
        import re
        match = re.search(r"try again in ([\d\.]+)s", response.text)
        if match:
            return max(float(match.group(1)) + 0.5, 1.0)
    except Exception:
        pass
    return fallback_delay


class GroqClient:
    def __init__(self):
        # httpx.AsyncClient is NOT created here — it must be created inside
        # an async context (running event loop). Call init() after construction.
        self._http: httpx.AsyncClient | None = None

        # Ordered list of API keys taken from settings at init() time.
        self._keys: list[str] = []

        # Per-key cooldown tracking: maps key → monotonic timestamp at which
        # the cooldown expires (None = available now).
        # Keys are stored as their *index* rather than the raw string in log
        # messages to avoid ever leaking credentials.
        self._cooldown_until: dict[str, float] = {}

        # Round-robin cursor: index into self._keys for the *next* call.
        self._rr_cursor: int = 0

    async def init(self):
        """Create the underlying httpx.AsyncClient. Must be called from an
        async context (e.g. FastAPI lifespan startup).

        The Authorization header is now set *per-request* (not baked into the
        client's default headers) so that key rotation works at call granularity.
        """
        self._keys = settings.parsed_groq_api_keys
        self._cooldown_until = {k: 0.0 for k in self._keys}  # 0.0 → available immediately
        self._rr_cursor = 0
        self._http = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            timeout=60,
        )
        logger.info(
            "GroqClient initialized with %d key(s).", len(self._keys)
        )

    def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError(
                "GroqClient not initialized — call init_groq() at startup"
            )
        return self._http

    def _next_available_key(self) -> tuple[str, int, float]:
        """Return ``(key, key_index, sleep_needed)`` for the next request.

        Rotation strategy:
          1. Scan keys starting from the current round-robin cursor; skip any
             whose cooldown has not yet expired.
          2. If at least one key is available: return it (no sleep, 0.0) and
             advance the cursor past it.
          3. If *every* key is in cooldown: return the key with the soonest
             expiry so the caller can sleep exactly until it becomes available,
             then use that key.

        ``sleep_needed`` is the number of seconds the caller should sleep
        *before* sending the request.  It is 0.0 when a key is immediately
        available.
        """
        now = time.monotonic()
        n = len(self._keys)

        # First pass: look for an available key starting from the cursor.
        for offset in range(n):
            idx = (self._rr_cursor + offset) % n
            key = self._keys[idx]
            if self._cooldown_until.get(key, 0.0) <= now:
                self._rr_cursor = (idx + 1) % n
                return key, idx, 0.0

        # All keys are cooling down — find the one with the soonest expiry.
        soonest_idx, soonest_key = min(
            enumerate(self._keys),
            key=lambda pair: self._cooldown_until.get(pair[1], 0.0),
        )
        sleep_needed = max(self._cooldown_until[soonest_key] - now, 0.0)
        self._rr_cursor = (soonest_idx + 1) % n
        return soonest_key, soonest_idx, sleep_needed

    def _mark_cooldown(self, key: str, key_idx: int, duration: float) -> None:
        """Put ``key`` into cooldown for ``duration`` seconds."""
        expires_at = time.monotonic() + duration
        self._cooldown_until[key] = expires_at
        logger.info(
            "Key index %d entered cooldown for %.2fs (monotonic expiry: %.3f).",
            key_idx,
            duration,
            expires_at,
        )

    async def chat(self, system: str, user: str, model: str, temperature: float = 0.2) -> str:
        http = self._ensure_http()
        backoffs = [2.0, 5.0, 10.0, 20.0]
        attempt = 0
        max_attempts = len(backoffs) + 1

        # When a non-429 error fires we want to retry the *same* key (not
        # rotate), so we pin it here.  None means "ask _next_available_key()".
        _pinned: tuple[str, int] | None = None

        while attempt < max_attempts:
            if _pinned is not None:
                # Non-429 retry: reuse the exact key that failed.
                key, key_idx = _pinned
                sleep_needed = 0.0
            else:
                key, key_idx, sleep_needed = self._next_available_key()

            # All keys are simultaneously cooling down → honour the backoff sleep
            # so we don't spin-loop and to make forward progress.
            if sleep_needed > 0.0:
                logger.warning(
                    "All %d key(s) cooling down. Sleeping %.2fs before retry "
                    "(attempt %d/%d, next key index %d)...",
                    len(self._keys),
                    sleep_needed,
                    attempt + 1,
                    max_attempts,
                    key_idx,
                )
                await asyncio.sleep(sleep_needed)

            logger.info(
                "chat() attempt %d/%d using key index %d.",
                attempt + 1,
                max_attempts,
                key_idx,
            )
            try:
                resp = await http.post(
                    "/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": temperature,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                if status == 429 and attempt < len(backoffs):
                    # Rate-limit on this key: record cooldown and rotate immediately.
                    delay = _extract_retry_delay(e.response, backoffs[attempt])
                    self._mark_cooldown(key, key_idx, delay)
                    logger.warning(
                        "Key index %d rate-limited (429). Rotating to next key immediately "
                        "(attempt %d/%d).",
                        key_idx,
                        attempt + 1,
                        len(backoffs),
                    )
                    _pinned = None  # clear pin → pick a fresh key next iteration
                    attempt += 1
                    continue  # No sleep — try the next key immediately.

                if status >= 500 and attempt < len(backoffs):
                    # Server error: backoff on the SAME key (not a rate-limit signal).
                    delay = _extract_retry_delay(e.response, backoffs[attempt])
                    logger.warning(
                        "Groq API server error %d (key index %d). Backing off for %.2fs "
                        "before retry (attempt %d/%d)...",
                        status,
                        key_idx,
                        delay,
                        attempt + 1,
                        len(backoffs),
                    )
                    await asyncio.sleep(delay)
                    _pinned = (key, key_idx)  # pin → retry same key
                    attempt += 1
                    continue

                logger.error("Groq API HTTP error %d: %s", status, e.response.text)
                raise RuntimeError(f"Groq API HTTP error {status}: {e.response.text}") from e

            except httpx.RequestError as e:
                if attempt < len(backoffs):
                    delay = backoffs[attempt]
                    logger.warning(
                        "Groq API request error (key index %d): %s. Retrying in %.2fs...",
                        key_idx,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    _pinned = (key, key_idx)  # pin → retry same key
                    attempt += 1
                    continue
                logger.error("Groq API request failed: %s", e)
                raise RuntimeError(f"Groq API request failed: {e}") from e

        # Exhausted all retries without a successful response.
        raise RuntimeError("Groq API: all retry attempts exhausted")


    async def chat_structured(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict,
        schema_name: str,
        temperature: float = 0.0,
    ) -> dict:
        """Like chat(), but requests Groq's strict structured JSON output.

        Uses response_format with json_schema to guarantee the response
        conforms to the provided JSON Schema. Returns the parsed dict.
        """
        # Groq's strict mode requires additionalProperties=false and rejects
        # the 'title' fields that Pydantic's model_json_schema() adds.
        schema = _prepare_schema_for_groq(schema)
        http = self._ensure_http()
        backoffs = [2.0, 5.0, 10.0, 20.0]
        attempt = 0
        max_attempts = len(backoffs) + 1

        # Same key-pinning logic as chat(): 500/network retries reuse the same
        # key; only 429 triggers rotation.
        _pinned: tuple[str, int] | None = None

        while attempt < max_attempts:
            if _pinned is not None:
                key, key_idx = _pinned
                sleep_needed = 0.0
            else:
                key, key_idx, sleep_needed = self._next_available_key()

            if sleep_needed > 0.0:
                logger.warning(
                    "All %d key(s) cooling down. Sleeping %.2fs before retry "
                    "(attempt %d/%d, next key index %d)...",
                    len(self._keys),
                    sleep_needed,
                    attempt + 1,
                    max_attempts,
                    key_idx,
                )
                await asyncio.sleep(sleep_needed)

            logger.info(
                "chat_structured() attempt %d/%d using key index %d.",
                attempt + 1,
                max_attempts,
                key_idx,
            )
            try:
                resp = await http.post(
                    "/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": temperature,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "schema": schema,
                                "strict": True,
                            },
                        },
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Groq strict mode guarantees valid JSON, but defense in depth
                    logger.error(
                        "Groq structured output was not valid JSON: %s", content[:200]
                    )
                    raise RuntimeError(
                        "Groq structured output was not valid JSON despite strict mode"
                    )

            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                if status == 429 and attempt < len(backoffs):
                    delay = _extract_retry_delay(e.response, backoffs[attempt])
                    self._mark_cooldown(key, key_idx, delay)
                    logger.warning(
                        "Key index %d rate-limited (429). Rotating to next key immediately "
                        "(attempt %d/%d).",
                        key_idx,
                        attempt + 1,
                        len(backoffs),
                    )
                    _pinned = None  # clear pin → pick a fresh key next iteration
                    attempt += 1
                    continue  # No sleep — try the next key immediately.

                if status >= 500 and attempt < len(backoffs):
                    delay = _extract_retry_delay(e.response, backoffs[attempt])
                    logger.warning(
                        "Groq API server error %d (key index %d). Backing off for %.2fs "
                        "before retry (attempt %d/%d)...",
                        status,
                        key_idx,
                        delay,
                        attempt + 1,
                        len(backoffs),
                    )
                    await asyncio.sleep(delay)
                    _pinned = (key, key_idx)  # pin → retry same key
                    attempt += 1
                    continue

                logger.error("Groq API HTTP error %d: %s", status, e.response.text)
                raise RuntimeError(f"Groq API HTTP error {status}: {e.response.text}") from e

            except httpx.RequestError as e:
                if attempt < len(backoffs):
                    delay = backoffs[attempt]
                    logger.warning(
                        "Groq API request error (key index %d): %s. Retrying in %.2fs...",
                        key_idx,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    _pinned = (key, key_idx)  # pin → retry same key
                    attempt += 1
                    continue
                logger.error("Groq API request failed: %s", e)
                raise RuntimeError(f"Groq API request failed: {e}") from e

        raise RuntimeError("Groq API: all retry attempts exhausted")

    async def aclose(self):
        if self._http is not None:
            await self._http.aclose()
            self._http = None


async def init_groq() -> GroqClient:
    """Create and initialize the module-level GroqClient singleton.
    Call from FastAPI lifespan startup, same pattern as init_pool()."""
    global _client
    _client = GroqClient()
    await _client.init()
    return _client


def get_groq_client() -> GroqClient:
    """Return the initialized GroqClient. Raises if init_groq() hasn't been called."""
    if _client is None:
        raise RuntimeError("Groq client not initialized — call init_groq() at startup")
    return _client


async def close_groq():
    """Shut down the GroqClient. Call from FastAPI lifespan shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
