import asyncpg
from pgvector.asyncpg import register_vector

from app.config import settings

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    """Create the Neon connection pool once, at app startup.

    Neon is serverless: connections can be recycled by the platform, so we
    keep the pool small and let asyncpg handle reconnects rather than
    holding many long-lived connections open.
    """
    global _pool

    async def _init_connection(conn: asyncpg.Connection):
        await register_vector(conn)

    _pool = await asyncpg.create_pool(
        dsn=settings.neon_database_url,
        min_size=1,
        max_size=5,
        init=_init_connection,
        command_timeout=30,
    )
    return _pool


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized — call init_pool() at startup")
    return _pool


async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
