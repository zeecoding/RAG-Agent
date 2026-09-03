"""Local embeddings via sentence-transformers — runs on CPU, no GPU or
external API needed. The model loads once at import time and stays in
memory; encoding a chunk takes well under a second on CPU.

BGE models (BAAI/bge-*) benefit from a query instruction prefix that
tells the model the text is a search query rather than a document passage.
This prefix is added ONLY at query time (embed), never during ingestion
(embed_batch), because the model was trained with asymmetric input:
queries get the prefix, passages don't.
"""
import asyncio

from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


# BGE models use an instruction prefix for queries (not for passages).
# Other models (e.g. all-MiniLM-L6-v2) don't need one.
_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
    if "bge" in settings.embedding_model.lower()
    else ""
)


def _encode_sync(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


async def embed(text: str) -> list[float]:
    """Embed a single query — applies the BGE instruction prefix."""
    prefixed = _QUERY_PREFIX + text
    result = await asyncio.to_thread(_encode_sync, [prefixed])
    return result[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed document passages — NO prefix, per BGE's asymmetric design."""
    return await asyncio.to_thread(_encode_sync, texts)
