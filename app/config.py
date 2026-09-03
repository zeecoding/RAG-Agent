from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neon_database_url: str

    groq_api_key: str
    groq_draft_model: str = "openai/gpt-oss-20b"
    groq_validate_model: str = "openai/gpt-oss-120b"

    embedding_model: str = "BAAI/bge-base-en-v1.5"  # 768 dims — must match sql/schema.sql
    embedding_dim: int = 768

    # DEPRECATED: No longer used — retrieval now uses Reciprocal Rank Fusion
    # (RRF) which combines ranks, not raw scores. Kept here so existing .env
    # files that set these values don't cause startup errors.
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    max_refine_attempts: int = 3
    top_k_chunks: int = 6

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Upload
    max_upload_size_mb: int = 50
    allowed_extensions: str = ".pdf,.docx,.xlsx,.csv,.txt,.md"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
