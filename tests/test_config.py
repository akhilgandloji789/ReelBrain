import pytest
from pathlib import Path
from config import Settings

def test_settings_load_and_embedding_config(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:fake_token")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123456789")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USERS", "111, 222, 333")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("TOPIC_RECIPE", "12")
    monkeypatch.setenv("TOPIC_TECH", "15")

    settings = Settings()
    assert settings.TELEGRAM_BOT_TOKEN == "12345:fake_token"
    assert settings.TELEGRAM_GROUP_CHAT_ID == -100123456789 or settings.TELEGRAM_GROUP_CHAT_ID == "-100123456789"
    assert settings.EMBEDDING_MODEL == "models/gemini-embedding-001"
    assert settings.EMBEDDING_DIM == 768
    assert settings.allowed_users == [111, 222, 333]
    assert settings.TEMP_DIR.exists()
    assert settings.DATA_DIR.exists()
    assert settings.topic_map.get("recipe") == 12
    assert settings.topic_map.get("tech") == 15
    assert settings.topic_map.get("other") is None

def test_settings_default_embeddings(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:fake_token")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123456789")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))

    settings = Settings()
    assert settings.EMBEDDING_MODEL == "models/gemini-embedding-001"
    assert settings.EMBEDDING_DIM == 768
    assert settings.allowed_users == []

def test_settings_missing_token_raises(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(Exception):
        Settings()

def test_settings_cleanup_temp_toggle(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:fake_token")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123456789")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
    monkeypatch.setenv("CLEANUP_TEMP", "false")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))

    settings = Settings()
    assert settings.CLEANUP_TEMP is False
