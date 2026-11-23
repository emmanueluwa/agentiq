from functools import lru_cache
from pathlib import Path
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    test_database_url: str | None = Field(
        default=None,
        alias="TEST_DATABASE_URL",
    )

    database_url: str = Field(
        default=os.getenv("DATABASE_URL"),
        alias="DATABASE_URL",
    )

    grok_api_key: str | None = Field(default=None, alias="GROK_API_KEY")
    grok_base_url: str = Field(
        default="https://api.x.ai/v1",
        alias="GROK_BASE_URL",
    )
    grok_model: str = Field(default="grok-4-1-fast", alias="GROK_MODEL")

    max_completion_tokens: int = Field(default=4096, alias="MAX_COMPLETION_TOKENS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    exa_api_key: str | None = Field(
        default=None,
        alias="EXA_API_KEY",
    )

    exa_base_url: str = Field(
        default="https://api.exa.ai",
        alias="EXA_BASE_URL",
    )

    research_max_results: int = Field(
        default=8,
        alias="RESEARCH_MAX_RESULTS",
    )

    research_min_results: int = Field(
        default=2,
        alias="RESEARCH_MIN_RESULTS",
    )

    research_request_timeout_seconds: float = Field(
        default=20.0,
        alias="RESEARCH_REQUEST_TIMEOUT_SECONDS",
    )

    frontend_origin: str = Field(
        default="http://localhost:5173",
        alias="FRONTEND_ORIGIN",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = Settings()
