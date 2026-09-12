from pathlib import Path
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_GROUP_CHAT_ID: int | str
    GEMINI_API_KEY: str
    ALLOWED_TELEGRAM_USERS: str = ""
    
    # Vector Embeddings
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIM: int = 768

    # Health & Canary
    CANARY_REEL_URL: str | None = None

    # Paths
    DATA_DIR: Path = Path("./data")
    TEMP_DIR: Path = Path("./temp")

    # Forum Topic Thread IDs
    TOPIC_RECIPE: int | None = None
    TOPIC_TECH: int | None = None
    TOPIC_WORKOUT: int | None = None
    TOPIC_IDEA: int | None = None
    TOPIC_TRAVEL: int | None = None
    TOPIC_FINANCE: int | None = None
    TOPIC_OTHER: int | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def model_post_init(self, __context: Any) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def allowed_users(self) -> list[int]:
        if not self.ALLOWED_TELEGRAM_USERS:
            return []
        users = []
        for part in self.ALLOWED_TELEGRAM_USERS.split(","):
            cleaned = part.strip()
            if cleaned.isdigit() or (cleaned.startswith("-") and cleaned[1:].isdigit()):
                users.append(int(cleaned))
        return users

    @property
    def topic_map(self) -> dict[str, int | None]:
        return {
            "recipe": self.TOPIC_RECIPE,
            "tech": self.TOPIC_TECH,
            "workout": self.TOPIC_WORKOUT,
            "idea": self.TOPIC_IDEA,
            "travel": self.TOPIC_TRAVEL,
            "finance": self.TOPIC_FINANCE,
            "other": self.TOPIC_OTHER,
        }


# Lazy global settings accessor
_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
