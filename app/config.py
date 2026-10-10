from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neon_database_url: str

    # Comma-separated Groq API keys, e.g. "gsk_abc,gsk_def,gsk_ghi".
    # A single key without commas works identically to the previous groq_api_key field.
    groq_api_keys: str
    groq_draft_model: str = "openai/gpt-oss-20b"
    groq_validate_model: str = "openai/gpt-oss-120b"

    @property
    def parsed_groq_api_keys(self) -> list[str]:
        """Parse groq_api_keys into a list, stripping whitespace and empty entries.

        Supports both single-key configs ("gsk_abc") and multi-key configs
        ("gsk_abc, gsk_def, gsk_ghi").  Order is preserved so callers can
        rely on a stable round-robin sequence.
        """
        keys = [k.strip() for k in self.groq_api_keys.split(",")]
        keys = [k for k in keys if k]
        if not keys:
            raise ValueError("groq_api_keys must contain at least one non-empty key")
        return keys

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

    # Supabase (Auth & Object Storage)
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_storage_bucket: str = "questionnaires"

    # Base URL of this FastAPI service, used for ONLYOFFICE callback URLs.
    # Example: https://api.myapp.com  (no trailing slash)
    api_base_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
