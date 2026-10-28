"""Settings from environment variables. Fails fast on unsafe production config."""
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Csv = Annotated[list[str], NoDecode]   # comma-separated env value -> list


def _csv(v):
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return v or []


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENV: str = "development"                # development | production
    DATABASE_URL: str                       # Supabase transaction pooler (port 6543)

    API_KEYS: Csv = Field(default_factory=list)    # server-to-server keys for the frontend backend
    ADMIN_KEYS: Csv = Field(default_factory=list)
    CORS_ORIGINS: Csv = Field(default_factory=list)
    ALLOWED_IMAGE_HOSTS: Csv = Field(default_factory=list)

    MODAL_APP: str = "turtle-pipeline"
    MODAL_CLASS: str = "Pipeline"

    ENABLE_DOCS: bool = True                # /docs and /openapi.json
    RATE_LIMIT_PER_UID: int = 10            # identify requests per uid per minute
    RATE_LIMIT_PER_KEY: int = 300           # identify requests per API key per minute
    MAX_BODY_BYTES: int = 32_768
    IDENTIFY_WAIT_SECONDS: int = 240        # POST /v1/identify waits this long for the result (GPU cold start ~1-2 min)

    _split = field_validator("API_KEYS", "ADMIN_KEYS", "CORS_ORIGINS", "ALLOWED_IMAGE_HOSTS",
                             mode="before")(_csv)

    @property
    def sqlalchemy_url(self) -> str:
        url = self.DATABASE_URL
        for prefix in ("postgresql+psycopg2://", "postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url

    @property
    def is_prod(self) -> bool:
        return self.ENV == "production"

    @model_validator(mode="after")
    def production_guards(self):
        if self.is_prod:
            problems = []
            if not self.API_KEYS:
                problems.append("API_KEYS is empty")
            if not self.ADMIN_KEYS:
                problems.append("ADMIN_KEYS is empty")
            if any(len(k) < 32 for k in self.API_KEYS + self.ADMIN_KEYS):
                problems.append("API/admin keys must be at least 32 characters")
            if not self.ALLOWED_IMAGE_HOSTS:
                problems.append("ALLOWED_IMAGE_HOSTS is empty")
            if "*" in self.CORS_ORIGINS:
                problems.append("CORS_ORIGINS must not be '*' in production")
            if problems:
                raise ValueError("unsafe production config: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
