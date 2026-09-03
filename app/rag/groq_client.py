"""Thin async client for Groq's OpenAI-compatible chat completions API.

Groq is a *hosted* inference API (you call their endpoint; they run the
model on their hardware) — not Ollama. No local or remote server to manage,
which is why this replaces the earlier Ollama client for a no-GPU setup.

Lifecycle follows the same pattern as db/pool.py:
  - init_groq()       → call from FastAPI lifespan startup
  - get_groq_client() → accessor, raises if not initialized
  - close_groq()      → call from FastAPI lifespan shutdown
"""
import asyncio
import json
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


class GroqClient:
    def __init__(self):
        # httpx.AsyncClient is NOT created here — it must be created inside
        # an async context (running event loop). Call init() after construction.
        self._http: httpx.AsyncClient | None = None

    async def init(self):
        """Create the underlying httpx.AsyncClient. Must be called from an
        async context (e.g. FastAPI lifespan startup)."""
        self._http = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            timeout=60,
        )

    def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError(
                "GroqClient not initialized — call init_groq() at startup"
            )
        return self._http

    async def chat(self, system: str, user: str, model: str, temperature: float = 0.2) -> str:
        http = self._ensure_http()
        backoffs = [1, 2, 4]
        for attempt in range(4):
            try:
                resp = await http.post(
                    "/chat/completions",
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
                if status == 429 or status >= 500:
                    if attempt < 3:
                        logger.warning(f"Groq API error ({status}). Retrying in {backoffs[attempt]}s...")
                        await asyncio.sleep(backoffs[attempt])
                        continue
                logger.error(f"Groq API HTTP error {status}: {e.response.text}")
                raise RuntimeError(f"Groq API HTTP error {status}: {e.response.text}") from e
            except httpx.RequestError as e:
                if attempt < 3:
                    logger.warning(f"Groq API request error: {e}. Retrying in {backoffs[attempt]}s...")
                    await asyncio.sleep(backoffs[attempt])
                    continue
                logger.error(f"Groq API request failed: {e}")
                raise RuntimeError(f"Groq API request failed: {e}") from e

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
        backoffs = [1, 2, 4]
        for attempt in range(4):
            try:
                resp = await http.post(
                    "/chat/completions",
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
                    logger.error(f"Groq structured output was not valid JSON: {content[:200]}")
                    raise RuntimeError(
                        "Groq structured output was not valid JSON despite strict mode"
                    )
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429 or status >= 500:
                    if attempt < 3:
                        logger.warning(f"Groq API error ({status}). Retrying in {backoffs[attempt]}s...")
                        await asyncio.sleep(backoffs[attempt])
                        continue
                logger.error(f"Groq API HTTP error {status}: {e.response.text}")
                raise RuntimeError(f"Groq API HTTP error {status}: {e.response.text}") from e
            except httpx.RequestError as e:
                if attempt < 3:
                    logger.warning(f"Groq API request error: {e}. Retrying in {backoffs[attempt]}s...")
                    await asyncio.sleep(backoffs[attempt])
                    continue
                logger.error(f"Groq API request failed: {e}")
                raise RuntimeError(f"Groq API request failed: {e}") from e

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
