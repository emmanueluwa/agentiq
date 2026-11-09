from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    grok_api_key: str | None = Field(default=None, alias="GROK_API_KEY")
    grok_base_url: str = Field(
        default="https://api.x.ai/v1",
        alias="GROK_BASE_URL",
    )
    grok_model: str = Field(default="grok-4-1-fast", alias="GROK_MODEL")

    max_completion_tokens: int = Field(default=4096, alias="MAX_COMPLETION_TOKENS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    frontend_origin: str = Field(
        default="http://localhost:5173",
        alias="FRONTEND_ORIGIN",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
