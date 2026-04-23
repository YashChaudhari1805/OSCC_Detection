"""
core/config.py
--------------
Centralised settings loaded from environment variables / .env file.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Model ---
    model_path: str = Field(default="oscc_model_rebuilt.keras")
    image_size: int = Field(default=224)
    max_upload_mb: int = Field(default=16)

    # --- LIME ---
    lime_num_samples: int = Field(default=1000)
    lime_num_features: int = Field(default=10)

    # --- Security ---
    api_keys_raw: str = Field(default="", alias="api_keys")
    rate_limit_per_minute: int = Field(default=30)
    allowed_origins: str = Field(default="*")

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    environment: str = Field(default="development")
    log_level: str = Field(default="info")
    secret_key: str = Field(default="change-me-in-production")

    @property
    def api_keys(self) -> list[str]:
        if not self.api_keys_raw:
            return []
        return [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]

    @property
    def auth_enabled(self) -> bool:
        return len(self.api_keys) > 0

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
