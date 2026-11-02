"""Runtime settings, read from environment variables (and `.env` in local dev)."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://papergraph:papergraph@localhost:5432/papergraph"
    anthropic_api_key: SecretStr | None = None
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
