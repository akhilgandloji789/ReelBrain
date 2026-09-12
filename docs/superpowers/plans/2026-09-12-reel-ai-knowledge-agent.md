# ReelMind: Personal AI Second Brain Implementation Plan (Master Edition)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ReelMind — a self-hosted, Telegram-native, multimodal personal second brain that turns saved short-form video content into structured, queryable, actionable knowledge anchored on **Memory → Evidence → Retrieval → Action**, featuring user intent capture, auditable provenance, pipeline state machines, interactive feedback loops, telemetry, and hackathon demo reliability.

**Architecture:** A decoupled, modular Python 3.11+ application featuring:
- **Resilient Ingestion:** Telegram link receiver parsing shortcodes and optional user intent ("why I saved this"), protected by an authorized user whitelist.
- **State Machine Pipeline:** Job states (`RECEIVED` → `DOWNLOADING` → `DOWNLOADED` → `ANALYZING` → `ANALYZED` → `EXTRACTING_FRAMES` → `PUBLISHING` → `INDEXING` → `COMPLETED`) with idempotency and tenacity exponential retries.
- **Multimodal AI & Real Confidence:** Gemini 2.0/2.5 Flash for vision and audio reasoning with nullable model-reported confidence scores (no fake 0.90 defaults), and configurable vector embeddings via `gemini-embedding-001` (768-dim).
- **Canonical SQLite Store & Derived FTS5:** WAL mode database with atomic transactions, synchronizing `reels`, `entities`, `embeddings`, and `reels_fts` derived full-text index.
- **Interactive UI & Feedback Loop:** Forum topic routing with inline keyboards (`[🛒 Grocery List]`, `[💻 Code]`, `[🔍 Ask]`, `[✏️ Edit]`, `[👍 Accurate]`, `[👎 Inaccurate]`), `/status` telemetry, and live `/canary` diagnostics.
- **Transactional Cleanup & Recovery:** Startup recovery job purging orphaned media, guaranteeing `<100MB` permanent local disk footprint.

**Tech Stack:** Python 3.11+, `python-telegram-bot>=21.0`, `google-genai>=0.1.0`, `yt-dlp>=2024.8.6`, `pydantic>=2.7.0`, `pydantic-settings>=2.2.0`, `apscheduler>=3.10.0`, `tenacity>=8.2.0`, `numpy>=1.26.0`, `pytest`, `pytest-asyncio`, `ffmpeg`.

**Spec:** [`docs/superpowers/specs/2026-09-12-reel-ai-knowledge-agent-design.md`](file:///c:/Akhil/Instagram/docs/superpowers/specs/2026-09-12-reel-ai-knowledge-agent-design.md)

## Global Constraints & Performance Budget
- **No Fake Defaults:** Confidence is nullable `float | None`; the UI displays verified timestamps rather than hardcoded 0.90 numbers.
- **Idempotency Guarantee:** Duplicate shortcodes are rejected before initiating downloads or publishing duplicate notes.
- **Data Sovereignty:** Full database is exportable to Markdown or JSON anytime via `/export`.
- **Performance Targets:** Processing latency ~10-25s for short reels. Max duration 90s. Max 4 keyframe snapshots.
- **Startup Recovery:** Ephemeral files in `temp/` are swept on startup, preventing orphaned media buildup.

---

## Database Schema (SQLite — `modules/storage.py`)

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS reels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    shortcode       TEXT UNIQUE NOT NULL,
    content_hash    TEXT,
    saved_at        TEXT NOT NULL,
    category        TEXT NOT NULL,
    title           TEXT,
    user_intent     TEXT,
    raw_transcript  TEXT,
    status          TEXT NOT NULL DEFAULT 'RECEIVED',
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_id         INTEGER NOT NULL REFERENCES reels(id) ON DELETE CASCADE,
    entity_type     TEXT NOT NULL,
    text            TEXT NOT NULL,
    start_ts        REAL NOT NULL,
    end_ts          REAL,
    confidence      REAL,                       -- NULLABLE, NO DEFAULT!
    frame_path      TEXT,
    edited_by_user  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embeddings (
    reel_id     INTEGER PRIMARY KEY REFERENCES reels(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS reels_fts USING fts5(
    title, user_intent, raw_transcript, entity_text,
    content='', tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS action_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_id     INTEGER REFERENCES reels(id),
    action_type TEXT NOT NULL,
    payload     TEXT,
    created_at  TEXT NOT NULL
);
```

---

### Task 1: Project Setup, Dependencies & Config with Configurable Embedding Settings

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `config.Settings` class:
  - `TELEGRAM_BOT_TOKEN: str`
  - `TELEGRAM_GROUP_CHAT_ID: int | str`
  - `GEMINI_API_KEY: str`
  - `ALLOWED_TELEGRAM_USERS: list[int]`
  - `EMBEDDING_MODEL: str` (defaults to `"models/gemini-embedding-001"`)
  - `EMBEDDING_DIM: int` (defaults to `768`)
  - `CANARY_REEL_URL: str | None`
  - `DATA_DIR: Path`
  - `TEMP_DIR: Path`
  - `topic_map: dict[str, int | None]`

- [ ] **Step 1: Write failing test for configuration loading and configurable embeddings**

```python
# tests/test_config.py
import pytest
from pathlib import Path
from config import Settings

def test_settings_load_and_embedding_config(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:fake_token")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123456789")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USERS", "111,222")
    
    settings = Settings()
    assert settings.TELEGRAM_BOT_TOKEN == "12345:fake_token"
    assert settings.EMBEDDING_MODEL == "models/gemini-embedding-001"
    assert settings.EMBEDDING_DIM == 768
    assert settings.allowed_users == [111, 222]
    assert settings.TEMP_DIR.exists()
    assert settings.DATA_DIR.exists()

def test_settings_missing_token_raises(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(Exception):
        Settings()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'config')

- [ ] **Step 3: Implement requirements.txt, .env.example, and config.py**

Create `requirements.txt`:
```text
python-telegram-bot>=21.0
google-genai>=0.1.0
yt-dlp>=2024.8.6
pydantic>=2.7.0
pydantic-settings>=2.2.0
apscheduler>=3.10.0
tenacity>=8.2.0
numpy>=1.26.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

Create `.env.example`:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_from_botfather
TELEGRAM_GROUP_CHAT_ID=-1001234567890
GEMINI_API_KEY=your_gemini_api_key_from_google_ai_studio
ALLOWED_TELEGRAM_USERS=12345678,98765432
EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_DIM=768
CANARY_REEL_URL=https://www.instagram.com/reel/C-stable123/
```

Create `config.py`:
```python
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    TELEGRAM_BOT_TOKEN: str = Field(..., description="Telegram bot token from @BotFather")
    TELEGRAM_GROUP_CHAT_ID: str | int = Field(..., description="Target Telegram Group / Channel ID")
    GEMINI_API_KEY: str = Field(..., description="Google AI Studio Gemini API Key")
    ALLOWED_TELEGRAM_USERS: str = Field(default="", description="Comma-separated Telegram user IDs allowed")
    
    EMBEDDING_MODEL: str = Field(default="models/gemini-embedding-001", description="Gemini embedding model name")
    EMBEDDING_DIM: int = Field(default=768, description="Vector dimension")
    CANARY_REEL_URL: str | None = Field(default=None, description="Stable public Reel URL for canary tests")

    TOPIC_RECIPES_THREAD_ID: int | None = None
    TOPIC_TECH_THREAD_ID: int | None = None
    TOPIC_FITNESS_THREAD_ID: int | None = None
    TOPIC_FINANCE_THREAD_ID: int | None = None
    TOPIC_BOOKS_THREAD_ID: int | None = None
    TOPIC_TRAVEL_THREAD_ID: int | None = None
    TOPIC_GENERAL_THREAD_ID: int | None = None

    DATA_DIR: Path = Path("data")
    TEMP_DIR: Path = Path("temp")

    def model_post_init(self, __context):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        (self.TEMP_DIR / "videos").mkdir(parents=True, exist_ok=True)
        (self.TEMP_DIR / "frames").mkdir(parents=True, exist_ok=True)
        (self.TEMP_DIR / "exports").mkdir(parents=True, exist_ok=True)

    @property
    def allowed_users(self) -> list[int]:
        if not self.ALLOWED_TELEGRAM_USERS:
            return []
        return [int(uid.strip()) for uid in self.ALLOWED_TELEGRAM_USERS.split(",") if uid.strip().isdigit()]

    @property
    def topic_map(self) -> dict[str, int | None]:
        return {
            "recipe": self.TOPIC_RECIPES_THREAD_ID,
            "tech": self.TOPIC_TECH_THREAD_ID,
            "workout": self.TOPIC_FITNESS_THREAD_ID,
            "finance": self.TOPIC_FINANCE_THREAD_ID,
            "idea": self.TOPIC_BOOKS_THREAD_ID,
            "travel": self.TOPIC_TRAVEL_THREAD_ID,
            "other": self.TOPIC_GENERAL_THREAD_ID,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add requirements.txt .env.example config.py tests/test_config.py
git commit -m "feat: implement configuration loader with configurable embeddings"
```

---

### Task 2: Canonical SQLite Database with State Machine, User Intent & Derived FTS5 Sync

**Files:**
- Create: `modules/__init__.py`
- Create: `modules/storage.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Produces: `ReelDatabase` class:
  - `add_reel(...) -> int`
  - `update_reel_status(reel_id: int, status: str, error_message: str | None = None) -> None`
  - `is_processed(shortcode: str) -> bool`
  - `get_reel_by_shortcode(shortcode: str) -> dict | None`
  - `get_reel_by_id(reel_id: int) -> dict | None`
  - `add_entities(reel_id: int, entities: list[dict]) -> None`
  - `get_entities(reel_id: int) -> list[dict]`
  - `update_entity(entity_id: int, new_text: str) -> tuple[bool, int]`
  - `store_embedding(reel_id: int, model: str, dim: int, vector: list[float]) -> None`
  - `get_all_embeddings() -> list[dict]`
  - `search_fts(query: str, limit: int = 5) -> list[dict]`
  - `log_action(reel_id: int | None, action_type: str, payload: dict) -> None`
  - `cleanup_orphaned_jobs(max_age_hours: int = 1) -> int`
  - `get_all_reels_with_entities() -> list[dict]`
  - `get_metrics_summary() -> dict`

- [ ] **Step 1: Write failing test for state machine, user intent, and derived FTS5 synchronization**

```python
# tests/test_storage.py
import pytest
from pathlib import Path
from modules.storage import ReelDatabase

def test_database_intent_and_fts_sync(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    shortcode = "C-xyz123"
    url = f"https://www.instagram.com/reel/{shortcode}/"
    
    reel_id = db.add_reel(
        url=url,
        shortcode=shortcode,
        category="recipe",
        title="10-Minute High-Protein Oats",
        user_intent="Try this for Sunday meal prep",
        raw_transcript="Oats and whey recipe"
    )
    
    assert db.get_reel_by_id(reel_id)["user_intent"] == "Try this for Sunday meal prep"
    assert db.get_reel_by_id(reel_id)["status"] == "RECEIVED"
    
    db.update_reel_status(reel_id, "COMPLETED")
    assert db.is_processed(shortcode) is True

    # Entities with nullable confidence (no fake default)
    entities = [
        {"entity_type": "ingredient", "text": "50g rolled oats", "start_ts": 8.0, "confidence": None},
        {"entity_type": "instruction", "text": "Mix with almond milk", "start_ts": 15.0, "confidence": 0.94}
    ]
    db.add_entities(reel_id, entities)
    
    stored = db.get_entities(reel_id)
    assert stored[0]["confidence"] is None
    assert stored[1]["confidence"] == 0.94

    # Search user intent via FTS5
    results = db.search_fts("meal prep")
    assert len(results) == 1
    assert results[0]["id"] == reel_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.storage')

- [ ] **Step 3: Implement modules/storage.py**

```python
# modules/storage.py
import json
import struct
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)

def unpack_vector(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))

class ReelDatabase:
    def __init__(self, db_path: Path | str = Path("data/reelminds.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reels (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    url             TEXT UNIQUE NOT NULL,
                    shortcode       TEXT UNIQUE NOT NULL,
                    content_hash    TEXT,
                    saved_at        TEXT NOT NULL,
                    category        TEXT NOT NULL,
                    title           TEXT,
                    user_intent     TEXT,
                    raw_transcript  TEXT,
                    status          TEXT NOT NULL DEFAULT 'RECEIVED',
                    error_message   TEXT
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    reel_id         INTEGER NOT NULL REFERENCES reels(id) ON DELETE CASCADE,
                    entity_type     TEXT NOT NULL,
                    text            TEXT NOT NULL,
                    start_ts        REAL NOT NULL,
                    end_ts          REAL,
                    confidence      REAL,
                    frame_path      TEXT,
                    edited_by_user  INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    reel_id     INTEGER PRIMARY KEY REFERENCES reels(id) ON DELETE CASCADE,
                    model       TEXT NOT NULL,
                    dim         INTEGER NOT NULL,
                    vector      BLOB NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS reels_fts USING fts5(
                    title, user_intent, raw_transcript, entity_text,
                    content='', tokenize='porter unicode61'
                );

                CREATE TABLE IF NOT EXISTS action_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    reel_id     INTEGER REFERENCES reels(id),
                    action_type TEXT NOT NULL,
                    payload     TEXT,
                    created_at  TEXT NOT NULL
                );
            """)
            conn.commit()

    def is_processed(self, shortcode: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM reels WHERE shortcode = ? AND status = 'COMPLETED'", (shortcode,))
            return cursor.fetchone() is not None

    def add_reel(
        self,
        url: str,
        shortcode: str,
        category: str,
        title: str,
        user_intent: str | None = None,
        raw_transcript: str = "",
        content_hash: str | None = None
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO reels (url, shortcode, content_hash, saved_at, category, title, user_intent, raw_transcript, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RECEIVED')
                """,
                (url, shortcode, content_hash, now, category, title, user_intent, raw_transcript)
            )
            reel_id = cursor.lastrowid
            self._sync_fts_for_reel(conn, reel_id)
            conn.commit()
            return reel_id

    def update_reel_status(self, reel_id: int, status: str, error_message: str | None = None) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE reels SET status = ?, error_message = ? WHERE id = ?", (status, error_message, reel_id))
            conn.commit()

    def add_entities(self, reel_id: int, entities: list[dict]) -> None:
        with self._get_connection() as conn:
            for ent in entities:
                conn.execute(
                    """
                    INSERT INTO entities (reel_id, entity_type, text, start_ts, end_ts, confidence, frame_path, edited_by_user)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reel_id,
                        ent.get("entity_type", "fact"),
                        ent.get("text", ""),
                        ent.get("start_ts", 0.0),
                        ent.get("end_ts"),
                        ent.get("confidence"),  # Can be None
                        ent.get("frame_path"),
                        ent.get("edited_by_user", 0)
                    )
                )
            self._sync_fts_for_reel(conn, reel_id)
            conn.commit()

    def _sync_fts_for_reel(self, conn: sqlite3.Connection, reel_id: int) -> None:
        reel = conn.execute("SELECT title, user_intent, raw_transcript FROM reels WHERE id = ?", (reel_id,)).fetchone()
        if not reel:
            return
        ents = conn.execute("SELECT text FROM entities WHERE reel_id = ?", (reel_id,)).fetchall()
        all_entity_text = " ".join([e["text"] for e in ents])

        conn.execute("DELETE FROM reels_fts WHERE rowid = ?", (reel_id,))
        conn.execute(
            "INSERT INTO reels_fts(rowid, title, user_intent, raw_transcript, entity_text) VALUES (?, ?, ?, ?, ?)",
            (reel_id, reel["title"] or "", reel["user_intent"] or "", reel["raw_transcript"] or "", all_entity_text)
        )

    def get_reel_by_shortcode(self, shortcode: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM reels WHERE shortcode = ?", (shortcode,)).fetchone()
            return dict(row) if row else None

    def get_reel_by_id(self, reel_id: int) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM reels WHERE id = ?", (reel_id,)).fetchone()
            return dict(row) if row else None

    def get_entities(self, reel_id: int) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM entities WHERE reel_id = ? ORDER BY start_ts ASC", (reel_id,))
            return [dict(row) for row in cursor.fetchall()]

    def update_entity(self, entity_id: int, new_text: str) -> tuple[bool, int]:
        with self._get_connection() as conn:
            ent = conn.execute("SELECT reel_id FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if not ent:
                return False, 0
            reel_id = ent["reel_id"]
            conn.execute("UPDATE entities SET text = ?, edited_by_user = 1 WHERE id = ?", (new_text, entity_id))
            self._sync_fts_for_reel(conn, reel_id)
            conn.commit()
            return True, reel_id

    def store_embedding(self, reel_id: int, model: str, dim: int, vector: list[float]) -> None:
        blob = pack_vector(vector)
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (reel_id, model, dim, vector, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (reel_id, model, dim, blob, now)
            )
            conn.commit()

    def get_all_embeddings(self) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.id, r.title, r.category, r.url, r.user_intent, e.model, e.dim, e.vector
                FROM embeddings e
                JOIN reels r ON e.reel_id = r.id
                """
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["embedding"] = unpack_vector(d["vector"])
                results.append(d)
            return results

    def search_fts(self, query: str, limit: int = 5) -> list[dict]:
        clean_query = query.replace('"', '""').strip()
        if not clean_query:
            return []
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.* FROM reels r
                JOIN reels_fts ON r.id = reels_fts.rowid
                WHERE reels_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (clean_query, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def log_action(self, reel_id: int | None, action_type: str, payload: dict) -> None:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO action_log (reel_id, action_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (reel_id, action_type, json.dumps(payload), now)
            )
            conn.commit()

    def get_recent_reels(self, days: int = 7) -> list[dict]:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM reels WHERE saved_at >= ? ORDER BY saved_at DESC", (since,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_reels_with_entities(self) -> list[dict]:
        with self._get_connection() as conn:
            reels = [dict(r) for r in conn.execute("SELECT * FROM reels ORDER BY saved_at DESC").fetchall()]
            for r in reels:
                ents = conn.execute("SELECT * FROM entities WHERE reel_id = ? ORDER BY start_ts ASC", (r["id"],)).fetchall()
                r["entities"] = [dict(e) for e in ents]
            return reels

    def get_metrics_summary(self) -> dict:
        with self._get_connection() as conn:
            total_reels = conn.execute("SELECT COUNT(*) FROM reels WHERE status = 'COMPLETED'").fetchone()[0]
            total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            total_actions = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]
            return {
                "total_reels": total_reels,
                "total_entities": total_entities,
                "total_embeddings": total_embeddings,
                "total_actions": total_actions
            }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/storage.py tests/test_storage.py
git commit -m "feat: implement canonical SQLite schema with intent, state machine and FTS sync"
```

---

### Task 3: Downloader Adapter with Circuit Breaker & Tenacity Retries

**Files:**
- Create: `modules/downloader.py`
- Create: `tests/test_downloader.py`

**Interfaces:**
- Produces: `Downloader` class:
  - `parse_input(raw_text: str) -> tuple[str | None, str | None, str | None]` (Returns shortcode, canonical_url, and optional user_intent)
  - `download_video(url: str, output_dir: Path) -> Path`

- [ ] **Step 1: Write failing test for shortcode and user intent parsing**

```python
# tests/test_downloader.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.downloader import Downloader

def test_parse_input_with_intent():
    downloader = Downloader()
    text = "https://www.instagram.com/reel/C-12345Abc/ Try this for Sunday meal prep"
    sid, canonical, intent = downloader.parse_input(text)
    assert sid == "C-12345Abc"
    assert canonical == "https://www.instagram.com/reel/C-12345Abc/"
    assert intent == "Try this for Sunday meal prep"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_downloader.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.downloader')

- [ ] **Step 3: Implement modules/downloader.py**

```python
# modules/downloader.py
import re
import time
from pathlib import Path
from yt_dlp import YoutubeDL
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

REEL_REGEX = re.compile(r'https?://(?:www\.)?instagram\.com/(?:reel|reels|p)/([a-zA-Z0-9_-]+)')

class DownloadError(Exception):
    pass

class CircuitBreakerOpen(Exception):
    pass

class Downloader:
    def __init__(self):
        self.consecutive_failures = 0
        self.circuit_open_until = 0

    def parse_input(self, raw_text: str) -> tuple[str | None, str | None, str | None]:
        match = REEL_REGEX.search(raw_text)
        if not match:
            return None, None, None
        shortcode = match.group(1)
        canonical = f"https://www.instagram.com/reel/{shortcode}/"
        
        # Everything outside the URL is user intent
        intent = raw_text.replace(match.group(0), "").strip()
        intent = intent if intent else None
        return shortcode, canonical, intent

    def _check_circuit(self):
        now = time.time()
        if now < self.circuit_open_until:
            remaining = int(self.circuit_open_until - now)
            raise CircuitBreakerOpen(f"Downloader circuit breaker active for {remaining}s due to consecutive failures.")

    def _record_success(self):
        self.consecutive_failures = 0

    def _record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.circuit_open_until = time.time() + 120

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type(DownloadError),
        reraise=True
    )
    def download_video(self, url: str, output_dir: Path) -> Path:
        self._check_circuit()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outtmpl = str(output_dir / "%(id)s.%(ext)s")
        ydl_opts = {
            "outtmpl": outtmpl,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info.get("id")
                expected_mp4 = output_dir / f"{video_id}.mp4"
                if expected_mp4.exists():
                    self._record_success()
                    return expected_mp4
                
                for f in output_dir.glob(f"{video_id}.*"):
                    self._record_success()
                    return f
                    
                raise DownloadError(f"Video file not found after download for {url}")
        except Exception as e:
            self._record_failure()
            raise DownloadError(f"Download failure on {url}: {str(e)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_downloader.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/downloader.py tests/test_downloader.py
git commit -m "feat: implement downloader with user intent parsing and circuit breaker"
```

---

### Task 4: Multimodal Analyzer with Real Confidence Scores & Configurable Embeddings

**Files:**
- Create: `modules/analyzer.py`
- Create: `tests/test_analyzer.py`

**Interfaces:**
- Produces:
  - `EntityItem` (`confidence: float | None = None` — no fake defaults!)
  - `ReelAnalysisOutput`
  - `ReelAnalyzer` class:
    - `analyze_video(video_path: Path) -> ReelAnalysisOutput`
    - `generate_embedding(text: str) -> list[float]` (Respects `config.EMBEDDING_DIM`)

- [ ] **Step 1: Write failing test for entity schemas with nullable confidence**

```python
# tests/test_analyzer.py
import pytest
from pathlib import Path
from modules.analyzer import ReelAnalysisOutput, EntityItem

def test_entity_nullable_confidence():
    item = EntityItem(entity_type="ingredient", text="50g Oats", start_ts=5.0)
    assert item.confidence is None  # Must be None, NOT hardcoded 0.90!
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzer.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.analyzer')

- [ ] **Step 3: Implement modules/analyzer.py**

```python
# modules/analyzer.py
import json
import time
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

CategoryType = Literal["recipe", "tech", "workout", "idea", "travel", "finance", "other"]

class EntityItem(BaseModel):
    entity_type: str = Field(description="Type: ingredient, instruction, code_snippet, tech_fact, exercise, idea_point")
    text: str = Field(description="Exact fact, instruction, measurement or code")
    start_ts: float = Field(description="Start time in seconds where this appears or is spoken")
    end_ts: float | None = Field(default=None, description="End time in seconds")
    confidence: float | None = Field(default=None, description="Model-reported confidence (nullable, no fake default)")

class ReelAnalysisOutput(BaseModel):
    title: str = Field(description="Short, crisp, descriptive title")
    category: CategoryType = Field(description="Category of the reel content")
    tldr: str = Field(description="1-2 sentence executive summary")
    entities: list[EntityItem] = Field(description="Extracted grounded claims, ingredients, instructions or code")
    keyframe_timestamps: list[float] = Field(description="2-4 timestamps for key visual evidence snapshots")

ANALYSIS_PROMPT = """
You are ReelMind, an expert multimodal knowledge extraction agent.
Analyze the video and audio of this Reel thoroughly.
Extract grounded knowledge where every fact, measurement, or instruction is linked to its exact video timestamp.
Do NOT fabricate confidence scores. If confident, provide a score (e.g. 0.95); otherwise leave confidence as null.
Classify the category (recipe, tech, workout, idea, travel, finance, other).
Identify 2 to 4 keyframe timestamps for high-res photo snapshots.
Return strict JSON matching the schema.
"""

class ReelAnalyzer:
    def __init__(self, api_key: str, embedding_model: str = "models/gemini-embedding-001", embedding_dim: int = 768):
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
    def analyze_video(self, video_path: Path) -> ReelAnalysisOutput:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        uploaded = self.client.files.upload(file=video_path)
        while uploaded.state == "PROCESSING":
            time.sleep(2)
            uploaded = self.client.files.get(name=uploaded.name)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[uploaded, "Index and extract structured knowledge with exact evidence timestamps."],
                config=types.GenerateContentConfig(
                    system_instruction=ANALYSIS_PROMPT,
                    response_mime_type="application/json",
                    response_schema=ReelAnalysisOutput,
                    temperature=0.2,
                )
            )
            data = json.loads(response.text)
            return ReelAnalysisOutput.model_validate(data)
        finally:
            try:
                self.client.files.delete(name=uploaded.name)
            except Exception:
                pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
    def generate_embedding(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )
        return response.embedding.values[:self.embedding_dim]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/analyzer.py tests/test_analyzer.py
git commit -m "feat: implement analyzer with nullable confidence and configurable embedding dimensions"
```

---

### Task 5: Keyframe Snapshot Extractor (`ffmpeg`)

**Files:**
- Create: `modules/frame_extractor.py`
- Create: `tests/test_frame_extractor.py`

**Interfaces:**
- Produces: `FrameExtractor` class:
  - `extract_frames(video_path: Path, timestamps: list[float], output_dir: Path) -> list[Path]`
  - `cleanup_files(paths: list[Path]) -> None`

- [ ] **Step 1: Write failing test for frame extraction**

```python
# tests/test_frame_extractor.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.frame_extractor import FrameExtractor

@patch("subprocess.run")
def test_extract_frames(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    video = tmp_path / "vid.mp4"
    video.write_text("dummy")
    
    extractor = FrameExtractor()
    frames = extractor.extract_frames(video, [5.0, 12.0], tmp_path / "frames")
    assert len(frames) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_extractor.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.frame_extractor')

- [ ] **Step 3: Implement modules/frame_extractor.py**

```python
# modules/frame_extractor.py
import subprocess
from pathlib import Path

class FrameExtractor:
    def extract_frames(self, video_path: Path, timestamps: list[float], output_dir: Path) -> list[Path]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_paths: list[Path] = []
        video_stem = video_path.stem

        for idx, ts in enumerate(timestamps[:4]):
            output_frame = output_dir / f"{video_stem}_frame_{idx}_{int(ts)}s.jpg"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(output_frame)
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                extracted_paths.append(output_frame)
            except Exception:
                continue
                
        return extracted_paths

    def cleanup_files(self, paths: list[Path]) -> None:
        for p in paths:
            try:
                p = Path(p)
                if p.exists():
                    p.unlink()
            except Exception:
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_extractor.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/frame_extractor.py tests/test_frame_extractor.py
git commit -m "feat: implement frame extractor module"
```

---

### Task 6: Telegram Publisher with Feedback Buttons & Truthful Provenance

**Files:**
- Create: `modules/publisher.py`
- Create: `tests/test_publisher.py`

**Interfaces:**
- Produces: `TelegramPublisher` class:
  - `format_note_html(analysis: ReelAnalysisOutput, original_url: str, user_intent: str | None = None) -> str`
  - `build_inline_keyboard(reel_id: int, category: str) -> InlineKeyboardMarkup`
  - `publish_reel(reel_id: int, analysis: ReelAnalysisOutput, frame_paths: list[Path], original_url: str, user_intent: str | None = None) -> int | None`

- [ ] **Step 1: Write failing test for HTML note formatting with intent and feedback buttons**

```python
# tests/test_publisher.py
import pytest
from modules.analyzer import ReelAnalysisOutput, EntityItem
from modules.publisher import TelegramPublisher

def test_format_note_html_truthful_confidence():
    publisher = TelegramPublisher(bot_token="fake", group_chat_id="-100")
    analysis = ReelAnalysisOutput(
        title="Quick Oats",
        category="recipe",
        tldr="Protein breakfast",
        entities=[
            EntityItem(entity_type="ingredient", text="50g Oats", start_ts=8.0, confidence=None),
            EntityItem(entity_type="instruction", text="Add milk", start_ts=15.0, confidence=0.94)
        ],
        keyframe_timestamps=[8.0]
    )
    html = publisher.format_note_html(analysis, "https://instagram.com/reel/C-test/", user_intent="Sunday meal prep")
    assert "💡 <b>Why You Saved This:</b> Sunday meal prep" in html
    assert "• 50g Oats <i>(00:08 | verified in video)</i>" in html
    assert "• Add milk <i>(00:15 | 94% conf)</i>" in html

    kb = publisher.build_inline_keyboard(reel_id=42, category="recipe")
    assert kb is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_publisher.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.publisher')

- [ ] **Step 3: Implement modules/publisher.py**

```python
# modules/publisher.py
import html
from pathlib import Path
from telegram import Bot, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from modules.analyzer import ReelAnalysisOutput

def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

class TelegramPublisher:
    def __init__(self, bot_token: str, group_chat_id: str | int, topic_map: dict[str, int | None] | None = None):
        self.bot = Bot(token=bot_token)
        self.group_chat_id = group_chat_id
        self.topic_map = topic_map or {}

    def get_thread_id(self, category: str) -> int | None:
        return self.topic_map.get(category.lower())

    def format_note_html(self, analysis: ReelAnalysisOutput, original_url: str, user_intent: str | None = None) -> str:
        safe_title = html.escape(analysis.title)
        safe_tldr = html.escape(analysis.tldr)

        intent_block = ""
        if user_intent:
            intent_block = f"💡 <b>Why You Saved This:</b> {html.escape(user_intent)}\n\n"

        entity_lines = []
        for e in analysis.entities:
            ts_str = format_timestamp(e.start_ts)
            conf_str = f"{int(e.confidence * 100)}% conf" if e.confidence is not None else "verified in video"
            entity_lines.append(f"• {html.escape(e.text)} <i>({ts_str} | {conf_str})</i>")

        body_block = "\n".join(entity_lines)
        safe_url = html.escape(original_url)

        return (
            f"🎬 <b>{safe_title}</b>\n\n"
            f"{intent_block}"
            f"📌 <b>TL;DR:</b>\n{safe_tldr}\n\n"
            f"⚡ <b>Key Evidence & Steps:</b>\n{body_block}\n\n"
            f"🏷️ <i>#{analysis.category}</i>\n\n"
            f'🔗 <b>Original Reel:</b> <a href="{safe_url}">{safe_url}</a>'
        )

    def build_inline_keyboard(self, reel_id: int, category: str) -> InlineKeyboardMarkup:
        rows = []
        if category == "recipe":
            rows.append([InlineKeyboardButton("🛒 Get Grocery List", callback_data=f"grocery:{reel_id}")])
        elif category == "tech":
            rows.append([InlineKeyboardButton("💻 Copy Code", callback_data=f"code:{reel_id}")])
        
        rows.append([
            InlineKeyboardButton("🔍 Ask", callback_data=f"ask:{reel_id}"),
            InlineKeyboardButton("✏️ Edit", callback_data=f"edit:{reel_id}")
        ])
        rows.append([
            InlineKeyboardButton("👍 Accurate", callback_data=f"thumb_up:{reel_id}"),
            InlineKeyboardButton("👎 Inaccurate", callback_data=f"thumb_down:{reel_id}")
        ])
        return InlineKeyboardMarkup(rows)

    async def publish_reel(self, reel_id: int, analysis: ReelAnalysisOutput, frame_paths: list[Path], original_url: str, user_intent: str | None = None) -> int | None:
        thread_id = self.get_thread_id(analysis.category)
        
        valid_frames = [p for p in frame_paths if Path(p).exists()]
        if valid_frames:
            media = []
            files_to_close = []
            try:
                for frame in valid_frames[:4]:
                    fp = open(frame, "rb")
                    files_to_close.append(fp)
                    media.append(InputMediaPhoto(media=fp))
                
                await self.bot.send_media_group(
                    chat_id=self.group_chat_id,
                    message_thread_id=thread_id,
                    media=media
                )
            finally:
                for fp in files_to_close:
                    fp.close()

        text_content = self.format_note_html(analysis, original_url, user_intent)
        reply_markup = self.build_inline_keyboard(reel_id, analysis.category)
        
        await self.bot.send_message(
            chat_id=self.group_chat_id,
            message_thread_id=thread_id,
            text=text_content,
            parse_mode="HTML",
            reply_markup=reply_markup,
            disable_web_page_preview=False
        )
        return thread_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_publisher.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/publisher.py tests/test_publisher.py
git commit -m "feat: implement publisher with feedback buttons, user intent and truthful provenance"
```

---

### Task 7: Ingestion Pipeline with State Machine & Startup Recovery (v1.0 MVP)

**Files:**
- Create: `modules/metrics.py`
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `ReelPipeline` class:
  - `process_url(raw_input: str, user_id: int | None = None) -> tuple[bool, str, int | None]`
  - `startup_recovery() -> None`

- [ ] **Step 1: Write failing test for state machine and pipeline execution**

```python
# tests/test_pipeline.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from pipeline import ReelPipeline
from modules.analyzer import ReelAnalysisOutput

@pytest.mark.asyncio
async def test_pipeline_state_machine_flow(tmp_path):
    settings = MagicMock()
    settings.TEMP_DIR = tmp_path
    settings.allowed_users = []
    
    mock_db = MagicMock()
    mock_db.is_processed.return_value = False
    mock_db.add_reel.return_value = 1
    
    mock_downloader = MagicMock()
    video_file = tmp_path / "test.mp4"
    video_file.write_text("dummy")
    mock_downloader.parse_input.return_value = ("C-test1", "https://instagram.com/reel/C-test1/", "Meal prep intent")
    mock_downloader.download_video.return_value = video_file
    
    mock_analyzer = MagicMock()
    output = ReelAnalysisOutput(
        title="Test Reel",
        category="tech",
        tldr="Summary",
        entities=[],
        keyframe_timestamps=[1.0]
    )
    mock_analyzer.analyze_video.return_value = output
    mock_analyzer.generate_embedding.return_value = [0.1, 0.2]
    
    mock_extractor = MagicMock()
    mock_extractor.extract_frames.return_value = []
    
    mock_publisher = AsyncMock()
    mock_publisher.publish_reel.return_value = 10
    
    pipeline = ReelPipeline(
        settings=settings,
        db=mock_db,
        downloader=mock_downloader,
        analyzer=mock_analyzer,
        extractor=mock_extractor,
        publisher=mock_publisher
    )
    
    success, msg, thread_id = await pipeline.process_url("https://instagram.com/reel/C-test1/ Meal prep intent")
    assert success is True
    assert "Test Reel" in msg
    mock_db.add_reel.assert_called_once()
    mock_db.update_reel_status.assert_called()
    assert not video_file.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'pipeline')

- [ ] **Step 3: Implement modules/metrics.py and pipeline.py**

Create `modules/metrics.py`:
```python
# modules/metrics.py
import time

class TelemetryMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.total_processed = 0
        self.total_errors = 0
        self.last_latency = 0.0

    def record_run(self, duration: float, success: bool):
        self.last_latency = duration
        if success:
            self.total_processed += 1
        else:
            self.total_errors += 1

    def get_status(self) -> dict:
        uptime_mins = int((time.time() - self.start_time) / 60)
        return {
            "uptime_mins": uptime_mins,
            "processed": self.total_processed,
            "errors": self.total_errors,
            "last_latency_sec": round(self.last_latency, 2)
        }
```

Create `pipeline.py`:
```python
# pipeline.py
import time
import asyncio
from pathlib import Path
from config import Settings
from modules.storage import ReelDatabase
from modules.downloader import Downloader
from modules.analyzer import ReelAnalyzer
from modules.frame_extractor import FrameExtractor
from modules.publisher import TelegramPublisher
from modules.metrics import TelemetryMetrics

class ReelPipeline:
    def __init__(
        self,
        settings: Settings,
        db: ReelDatabase | None = None,
        downloader: Downloader | None = None,
        analyzer: ReelAnalyzer | None = None,
        extractor: FrameExtractor | None = None,
        publisher: TelegramPublisher | None = None,
        metrics: TelemetryMetrics | None = None,
    ):
        self.settings = settings
        self.db = db or ReelDatabase(settings.DATA_DIR / "reelminds.db")
        self.downloader = downloader or Downloader()
        self.analyzer = analyzer or ReelAnalyzer(
            api_key=settings.GEMINI_API_KEY,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dim=settings.EMBEDDING_DIM
        )
        self.extractor = extractor or FrameExtractor()
        self.publisher = publisher or TelegramPublisher(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            group_chat_id=settings.TELEGRAM_GROUP_CHAT_ID,
            topic_map=settings.topic_map
        )
        self.metrics = metrics or TelemetryMetrics()
        self._lock = asyncio.Lock()

    def startup_recovery(self) -> None:
        """Purge orphaned temporary files on startup, maintaining <100MB disk footprint."""
        temp_dir = self.settings.TEMP_DIR
        now = time.time()
        for p in temp_dir.rglob("*"):
            if p.is_file() and (now - p.stat().st_mtime > 3600):  # older than 1 hr
                try:
                    p.unlink()
                except Exception:
                    pass

    async def process_url(self, raw_input: str, user_id: int | None = None) -> tuple[bool, str, int | None]:
        start_time = time.time()
        if user_id is not None and self.settings.allowed_users and user_id not in self.settings.allowed_users:
            return False, "⛔ Unauthorized user. Access restricted by administrator.", None

        async with self._lock:
            shortcode, canonical_url, user_intent = self.downloader.parse_input(raw_input)
            if not shortcode or not canonical_url:
                return False, "Not a valid Instagram Reel or Post URL.", None

            # Idempotency check
            if self.db.is_processed(shortcode):
                existing = self.db.get_reel_by_shortcode(shortcode)
                cat = existing.get("category", "General") if existing else "General"
                title = existing.get("title", "Reel") if existing else "Reel"
                return True, f"⚠️ Already saved under <b>#{cat}</b>:\n📌 <i>{title}</i>", None

            video_dir = self.settings.TEMP_DIR / "videos"
            frames_dir = self.settings.TEMP_DIR / "frames"
            video_file: Path | None = None
            frame_paths: list[Path] = []
            reel_id: int | None = None

            try:
                # State: RECEIVED -> register in DB
                reel_id = self.db.add_reel(
                    url=canonical_url,
                    shortcode=shortcode,
                    category="other",
                    title="Processing...",
                    user_intent=user_intent
                )

                # State: DOWNLOADING
                self.db.update_reel_status(reel_id, "DOWNLOADING")
                video_file = self.downloader.download_video(canonical_url, output_dir=video_dir)

                # State: DOWNLOADED -> ANALYZING
                self.db.update_reel_status(reel_id, "ANALYZING")
                analysis = self.analyzer.analyze_video(video_file)

                # State: ANALYZED -> EXTRACTING_FRAMES
                self.db.update_reel_status(reel_id, "EXTRACTING_FRAMES")
                frame_paths = self.extractor.extract_frames(
                    video_path=video_file,
                    timestamps=analysis.keyframe_timestamps,
                    output_dir=frames_dir
                )

                # Generate Embedding
                embedding: list[float] | None = None
                try:
                    embed_text = f"{analysis.title}\n{user_intent or ''}\n{analysis.tldr}\n" + " ".join([e.text for e in analysis.entities])
                    embedding = self.analyzer.generate_embedding(embed_text)
                except Exception:
                    pass

                # State: PUBLISHING
                self.db.update_reel_status(reel_id, "PUBLISHING")
                thread_id = await self.publisher.publish_reel(
                    reel_id=reel_id,
                    analysis=analysis,
                    frame_paths=frame_paths,
                    original_url=canonical_url,
                    user_intent=user_intent
                )

                # State: INDEXING
                self.db.update_reel_status(reel_id, "INDEXING")
                self.db.add_reel(
                    url=canonical_url,
                    shortcode=shortcode,
                    category=analysis.category,
                    title=analysis.title,
                    user_intent=user_intent,
                    raw_transcript=analysis.tldr
                )
                self.db.add_entities(reel_id, [e.model_dump() for e in analysis.entities])

                if embedding:
                    self.db.store_embedding(
                        reel_id,
                        model=self.settings.EMBEDDING_MODEL,
                        dim=self.settings.EMBEDDING_DIM,
                        vector=embedding
                    )

                # State: COMPLETED
                self.db.update_reel_status(reel_id, "COMPLETED")

                elapsed = time.time() - start_time
                self.metrics.record_run(elapsed, success=True)
                return True, f"✅ Indexed <b>{analysis.title}</b> under <i>#{analysis.category}</i> ({round(elapsed, 1)}s)", thread_id

            except Exception as e:
                if reel_id:
                    self.db.update_reel_status(reel_id, "FAILED", error_message=str(e))
                elapsed = time.time() - start_time
                self.metrics.record_run(elapsed, success=False)
                return False, f"⚠️ Failed to process reel: {str(e)}", None

            finally:
                if video_file and video_file.exists():
                    try:
                        video_file.unlink()
                    except Exception:
                        pass
                if frame_paths:
                    self.extractor.cleanup_files(frame_paths)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/metrics.py pipeline.py tests/test_pipeline.py
git commit -m "feat: implement state machine pipeline with startup recovery and intent support"
```

---

### Task 8: Hybrid Search Engine & Grounded Conversational `/ask` (v1.1)

**Files:**
- Create: `modules/search.py`
- Create: `tests/test_search.py`

**Interfaces:**
- Produces: `SearchEngine` class:
  - `hybrid_search(query: str, top_k: int = 5) -> list[dict]`
  - `answer_conversational_query(query: str) -> str`

- [ ] **Step 1: Write failing test for search engine**

```python
# tests/test_search.py
import pytest
from unittest.mock import MagicMock
from modules.search import SearchEngine, cosine_similarity

def test_cosine_similarity():
    assert pytest.approx(cosine_similarity([1.0, 0.0], [1.0, 0.0])) == 1.0
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0])) == 0.0

def test_hybrid_search():
    mock_db = MagicMock()
    mock_analyzer = MagicMock()
    mock_analyzer.generate_embedding.return_value = [1.0, 0.0]
    
    mock_db.search_fts.return_value = [{"id": 1, "title": "Mobility Routine", "category": "workout", "url": "u1"}]
    mock_db.get_all_embeddings.return_value = [{"id": 1, "title": "Mobility Routine", "category": "workout", "url": "u1", "embedding": [0.95, 0.05]}]
    
    engine = SearchEngine(db=mock_db, analyzer=mock_analyzer)
    results = engine.hybrid_search("pre gym mobility")
    assert len(results) > 0
    assert results[0]["title"] == "Mobility Routine"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.search')

- [ ] **Step 3: Implement modules/search.py**

```python
# modules/search.py
import numpy as np
from modules.storage import ReelDatabase
from modules.analyzer import ReelAnalyzer

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))

class SearchEngine:
    def __init__(self, db: ReelDatabase, analyzer: ReelAnalyzer):
        self.db = db
        self.analyzer = analyzer

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        fts_matches = self.db.search_fts(query, limit=top_k)
        seen_ids = {m["id"] for m in fts_matches}
        ranked_results = list(fts_matches)

        try:
            query_embedding = self.analyzer.generate_embedding(query)
            candidates = self.db.get_all_embeddings()
            scored = []
            for c in candidates:
                sim = cosine_similarity(query_embedding, c["embedding"])
                scored.append((sim, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            
            for sim, candidate in scored[:top_k]:
                if candidate["id"] not in seen_ids:
                    seen_ids.add(candidate["id"])
                    ranked_results.append(candidate)
        except Exception:
            pass

        return ranked_results[:top_k]

    def answer_conversational_query(self, query: str) -> str:
        matches = self.hybrid_search(query, top_k=3)
        if not matches:
            return "🔍 I couldn't find any saved reels matching your question."

        context_lines = []
        for idx, m in enumerate(matches, 1):
            ents = self.db.get_entities(m["id"])
            ent_summary = "; ".join([
                f"{e['text']} ({int(e['start_ts'])}s" + (f", {int(e['confidence']*100)}% conf)" if e['confidence'] else ")")
                for e in ents[:5]
            ])
            intent_str = f"Why saved: {m['user_intent']}\n" if m.get("user_intent") else ""
            context_lines.append(
                f"[{idx}] Title: {m.get('title')}\n"
                f"{intent_str}"
                f"Category: {m.get('category')}\n"
                f"Evidence: {ent_summary}\n"
                f"Source: {m.get('url')}"
            )
        context_str = "\n\n".join(context_lines)

        prompt = f"""
        You are ReelMind personal AI assistant. A user is asking:
        "{query}"

        Here is the relevant retrieved knowledge with evidence timestamps:
        {context_str}

        Answer the user's question directly and concisely in Telegram HTML. You MUST cite the specific evidence timestamps and include the original reel link. If confidence is present, cite it; otherwise say 'verified in video'. Refuse to answer from facts not present in the retrieved set.
        """
        response = self.analyzer.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_search.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/search.py tests/test_search.py
git commit -m "feat: implement hybrid search and conversational /ask with grounded provenance"
```

---

### Task 9: Action Hooks, Feedback Logging, Atomic Re-Embedding & Export (v1.2)

**Files:**
- Create: `modules/actions.py`
- Create: `tests/test_actions.py`

**Interfaces:**
- Produces: `ActionHandler` class:
  - `generate_grocery_list(reel_id: int) -> str`
  - `generate_code_block(reel_id: int) -> str`
  - `apply_edit(entity_id: int, new_text: str) -> tuple[bool, str]`
  - `record_feedback(reel_id: int, thumb: str) -> str`
  - `export_markdown(output_path: Path) -> Path`
  - `export_json(output_path: Path) -> Path`

- [ ] **Step 1: Write failing test for feedback logging, actions and export**

```python
# tests/test_actions.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from modules.actions import ActionHandler

def test_grocery_and_feedback():
    mock_db = MagicMock()
    mock_db.get_entities.return_value = [{"entity_type": "ingredient", "text": "50g Oats"}]
    handler = ActionHandler(db=mock_db, analyzer=MagicMock())
    
    grocery = handler.generate_grocery_list(reel_id=1)
    assert "• [ ] 50g Oats" in grocery
    
    msg = handler.record_feedback(reel_id=1, thumb="up")
    assert "Thanks for the feedback" in msg
    mock_db.log_action.assert_called_with(1, "feedback_thumb_up", {"vote": "up"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_actions.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.actions')

- [ ] **Step 3: Implement modules/actions.py**

```python
# modules/actions.py
import json
import html
from pathlib import Path
from modules.storage import ReelDatabase
from modules.analyzer import ReelAnalyzer

class ActionHandler:
    def __init__(self, db: ReelDatabase, analyzer: ReelAnalyzer):
        self.db = db
        self.analyzer = analyzer

    def generate_grocery_list(self, reel_id: int) -> str:
        entities = self.db.get_entities(reel_id)
        ingredients = [e["text"] for e in entities if e["entity_type"] == "ingredient"]
        if not ingredients:
            return "⚠️ No ingredients found for this reel."
        
        seen = set()
        deduped = []
        for ing in ingredients:
            if ing.lower() not in seen:
                seen.add(ing.lower())
                deduped.append(ing)

        lines = ["🛒 <b>Grocery Checklist:</b>\n"]
        for item in deduped:
            lines.append(f"• [ ] {html.escape(item)}")
        
        self.db.log_action(reel_id, "grocery", {"items": deduped})
        return "\n".join(lines)

    def generate_code_block(self, reel_id: int) -> str:
        entities = self.db.get_entities(reel_id)
        snippets = [e["text"] for e in entities if e["entity_type"] == "code_snippet"]
        if not snippets:
            return "⚠️ No code snippets found for this reel."

        lines = ["💻 <b>Extracted Code:</b>\n"]
        for s in snippets:
            lines.append(f"<code>{html.escape(s)}</code>\n")
        
        self.db.log_action(reel_id, "code", {"snippets": snippets})
        return "\n".join(lines)

    def record_feedback(self, reel_id: int, thumb: str) -> str:
        action_type = "feedback_thumb_up" if thumb == "up" else "feedback_thumb_down"
        self.db.log_action(reel_id, action_type, {"vote": thumb})
        if thumb == "up":
            return "👍 Thanks for the feedback! Marked as accurate."
        return "👎 Noted as inaccurate. You can tap [✏️ Edit] to correct any wrong details."

    def apply_edit(self, entity_id: int, new_text: str) -> tuple[bool, str]:
        ok, reel_id = self.db.update_entity(entity_id, new_text)
        if not ok:
            return False, f"Entity {entity_id} not found."

        try:
            reel = self.db.get_reel_by_id(reel_id)
            ents = self.db.get_entities(reel_id)
            combined_text = f"{reel['title']}\n{reel['user_intent'] or ''}\n" + " ".join([e["text"] for e in ents])
            new_embedding = self.analyzer.generate_embedding(combined_text)
            self.db.store_embedding(
                reel_id,
                model=self.analyzer.embedding_model,
                dim=self.analyzer.embedding_dim,
                vector=new_embedding
            )
        except Exception:
            pass

        self.db.log_action(reel_id, "edit", {"entity_id": entity_id, "new_text": new_text})
        return True, f"✅ Entity {entity_id} updated & vector index refreshed."

    def export_markdown(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reels = self.db.get_all_reels_with_entities()

        lines = ["# ReelMind Knowledge Export\n"]
        for r in reels:
            lines.append(f"## {r.get('title', 'Untitled')}")
            if r.get("user_intent"):
                lines.append(f"- **Why I saved this:** {r['user_intent']}")
            lines.append(f"- **Category:** #{r.get('category')}")
            lines.append(f"- **Source:** {r.get('url')}")
            lines.append(f"- **Summary:** {r.get('raw_transcript')}\n")
            lines.append("### Grounded Evidence:")
            for e in r.get("entities", []):
                conf_str = f", {int(e['confidence']*100)}% conf" if e.get("confidence") is not None else ""
                lines.append(f"- [{int(e['start_ts'])}s] {e['text']} ({e['entity_type']}{conf_str})")
            lines.append("\n---\n")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def export_json(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reels = self.db.get_all_reels_with_entities()
        output_path.write_text(json.dumps(reels, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_actions.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/actions.py tests/test_actions.py
git commit -m "feat: implement action hooks, feedback voting, re-embedding edit and export"
```

---

### Task 10: Sunday Review & Live Canary Health Check (v1.3)

**Files:**
- Create: `modules/scheduler.py`
- Create: `tests/test_scheduler.py`

**Interfaces:**
- Produces: `SchedulerService` class:
  - `build_weekly_digest() -> str | None`
  - `run_canary_test() -> tuple[bool, str]`
  - `start()`

- [ ] **Step 1: Write failing test for scheduler and live canary**

```python
# tests/test_scheduler.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from modules.scheduler import SchedulerService

@pytest.mark.asyncio
async def test_digest_and_canary():
    mock_db = MagicMock()
    mock_db.get_recent_reels.return_value = [{"title": "Oats", "category": "recipe", "url": "url1"}]
    mock_pipeline = MagicMock()
    mock_pipeline.process_url = AsyncMock(return_value=(True, "Success", 1))

    service = SchedulerService(
        db=mock_db,
        bot=MagicMock(),
        chat_id="-100",
        pipeline=mock_pipeline,
        canary_url="https://instagram.com/reel/test/"
    )
    digest = service.build_weekly_digest()
    assert "🧠 <b>YOUR REELMIND — SUNDAY REVIEW</b>" in digest
    
    ok, _ = await service.run_canary_test()
    assert ok is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scheduler.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.scheduler')

- [ ] **Step 3: Implement modules/scheduler.py**

```python
# modules/scheduler.py
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot
from modules.storage import ReelDatabase

logger = logging.getLogger("scheduler")

class SchedulerService:
    def __init__(self, db: ReelDatabase, bot: Bot, chat_id: str | int, pipeline=None, canary_url: str | None = None):
        self.db = db
        self.bot = bot
        self.chat_id = chat_id
        self.pipeline = pipeline
        self.canary_url = canary_url
        self.scheduler = AsyncIOScheduler()

    def build_weekly_digest(self) -> str | None:
        recent = self.db.get_recent_reels(days=7)
        if not recent:
            return None

        recipes = [r for r in recent if r["category"] == "recipe"]
        workouts = [r for r in recent if r["category"] == "workout"]
        tech = [r for r in recent if r["category"] == "tech"]
        others = [r for r in recent if r not in recipes + workouts + tech]

        lines = ["🧠 <b>YOUR REELMIND — SUNDAY REVIEW</b>\n"]
        if recipes:
            lines.append(f"🍳 <b>{len(recipes)} recipe{'s' if len(recipes) > 1 else ''} saved:</b>")
            for r in recipes[:3]:
                lines.append(f"• <a href='{r['url']}'>{r['title']}</a>")
            lines.append("")
        if workouts:
            lines.append(f"🏋️ <b>{len(workouts)} workout{'s' if len(workouts) > 1 else ''} saved:</b>")
            for w in workouts[:3]:
                lines.append(f"• <a href='{w['url']}'>{w['title']}</a>")
            lines.append("")
        if tech:
            lines.append(f"💻 <b>{len(tech)} tech idea{'s' if len(tech) > 1 else ''} saved:</b>")
            for t in tech[:3]:
                lines.append(f"• <a href='{t['url']}'>{t['title']}</a>")
            lines.append("")
        if others:
            lines.append(f"💡 <b>{len(others)} other discovery items saved.</b>\n")

        lines.append("<i>Ask me anything about these with /ask!</i>")
        return "\n".join(lines)

    async def send_weekly_digest(self):
        digest = self.build_weekly_digest()
        if digest:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=digest,
                parse_mode="HTML",
                disable_web_page_preview=True
            )

    async def run_canary_test(self) -> tuple[bool, str]:
        if not self.pipeline or not self.canary_url:
            return True, "No canary URL configured."
        try:
            success, msg, _ = await self.pipeline.process_url(self.canary_url)
            if not success:
                alert_text = f"🚨 <b>Canary Alert:</b> Pipeline health check failed!\nDetails: {msg}"
                await self.bot.send_message(chat_id=self.chat_id, text=alert_text, parse_mode="HTML")
                return False, msg
            return True, "Canary health check passed."
        except Exception as e:
            alert_text = f"🚨 <b>Canary Alert:</b> Exception during health check:\n{str(e)}"
            await self.bot.send_message(chat_id=self.chat_id, text=alert_text, parse_mode="HTML")
            return False, str(e)

    def start(self):
        self.scheduler.add_job(self.send_weekly_digest, "cron", day_of_week="sun", hour=9, minute=0)
        if self.canary_url:
            self.scheduler.add_job(self.run_canary_test, "cron", day_of_week="tue", hour=3, minute=0)
        self.scheduler.start()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scheduler.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/scheduler.py tests/test_scheduler.py
git commit -m "feat: implement Sunday digest and canary test scheduler"
```

---

### Task 11: Telegram Poller Daemon, Inline Handlers, Telemetry & Documentation

**Files:**
- Create: `main.py`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `README.md`

- [ ] **Step 1: Implement main.py connecting callbacks, commands, telemetry, startup recovery and CLI runner**

```python
# main.py
import argparse
import asyncio
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters
)
from config import Settings
from pipeline import ReelPipeline
from modules.search import SearchEngine
from modules.actions import ActionHandler
from modules.scheduler import SchedulerService

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("reelmind")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
        
    text = update.message.text.strip()
    user_id = update.effective_user.id if update.effective_user else None
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    search_engine: SearchEngine = context.application.bot_data["search_engine"]

    if "instagram.com" in text:
        status_msg = await update.message.reply_text("⏳ Processing reel... Analyzing video & audio with Gemini...")
        success, result_text, thread_id = await pipeline.process_url(text, user_id=user_id)
        await status_msg.edit_text(result_text, parse_mode="HTML")
    else:
        answer = search_engine.answer_conversational_query(text)
        await update.message.reply_text(answer, parse_mode="HTML")

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    actions: ActionHandler = context.application.bot_data["actions"]

    if data.startswith("grocery:"):
        reel_id = int(data.split(":")[1])
        grocery_text = actions.generate_grocery_list(reel_id)
        await query.message.reply_text(grocery_text, parse_mode="HTML")
    elif data.startswith("code:"):
        reel_id = int(data.split(":")[1])
        code_text = actions.generate_code_block(reel_id)
        await query.message.reply_text(code_text, parse_mode="HTML")
    elif data.startswith("thumb_up:"):
        reel_id = int(data.split(":")[1])
        msg = actions.record_feedback(reel_id, "up")
        await query.message.reply_text(msg)
    elif data.startswith("thumb_down:"):
        reel_id = int(data.split(":")[1])
        msg = actions.record_feedback(reel_id, "down")
        await query.message.reply_text(msg)
    elif data.startswith("ask:"):
        await query.message.reply_text("💬 Reply with <code>/ask &lt;your question&gt;</code>", parse_mode="HTML")
    elif data.startswith("edit:"):
        await query.message.reply_text("✏️ Use <code>/edit &lt;entity_id&gt; &lt;new text&gt;</code> to correct any item.", parse_mode="HTML")

async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    metrics = pipeline.metrics.get_status()
    db_metrics = pipeline.db.get_metrics_summary()

    msg = (
        f"📊 <b>ReelMind Telemetry & Status:</b>\n\n"
        f"⏱️ <b>Uptime:</b> {metrics['uptime_mins']} minutes\n"
        f"⚡ <b>Last Latency:</b> {metrics['last_latency_sec']}s\n"
        f"✅ <b>Total Indexed Reels:</b> {db_metrics['total_reels']}\n"
        f"🧬 <b>Total Grounded Entities:</b> {db_metrics['total_entities']}\n"
        f"🧠 <b>Total Embeddings:</b> {db_metrics['total_embeddings']}\n"
        f"🛠️ <b>Action Invocations:</b> {db_metrics['total_actions']}\n"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def handle_canary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    scheduler: SchedulerService = context.application.bot_data["scheduler"]
    status_msg = await update.message.reply_text("🧪 Running live Canary health diagnostic...")
    ok, details = await scheduler.run_canary_test()
    if ok:
        await status_msg.edit_text("✅ <b>Canary Check Passed!</b>\nTelegram, SQLite, Gemini, and Downloader are operational.")
    else:
        await status_msg.edit_text(f"🚨 <b>Canary Check Failed!</b>\n{details}")

async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: <code>/ask what was the recipe with oats?</code>", parse_mode="HTML")
        return
    query = " ".join(context.args)
    search_engine: SearchEngine = context.application.bot_data["search_engine"]
    answer = search_engine.answer_conversational_query(query)
    await update.message.reply_text(answer, parse_mode="HTML")

async def handle_grocery(update: Update, context: ContextTypes.DEFAULT_TYPE):
    actions: ActionHandler = context.application.bot_data["actions"]
    if context.args:
        try:
            reel_id = int(context.args[0])
            msg = actions.generate_grocery_list(reel_id)
            await update.message.reply_text(msg, parse_mode="HTML")
            return
        except ValueError:
            pass
    await update.message.reply_text("Usage: <code>/grocery &lt;reel_id&gt;</code>", parse_mode="HTML")

async def handle_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    actions: ActionHandler = context.application.bot_data["actions"]
    if context.args:
        try:
            reel_id = int(context.args[0])
            msg = actions.generate_code_block(reel_id)
            await update.message.reply_text(msg, parse_mode="HTML")
            return
        except ValueError:
            pass
    await update.message.reply_text("Usage: <code>/code &lt;reel_id&gt;</code>", parse_mode="HTML")

async def handle_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    actions: ActionHandler = context.application.bot_data["actions"]
    if len(context.args) >= 2:
        try:
            entity_id = int(context.args[0])
            new_text = " ".join(context.args[1:])
            ok, msg = actions.apply_edit(entity_id, new_text)
            await update.message.reply_text(msg)
            return
        except ValueError:
            pass
    await update.message.reply_text("Usage: <code>/edit &lt;entity_id&gt; &lt;corrected text&gt;</code>", parse_mode="HTML")

async def handle_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fmt = context.args[0].lower() if context.args else "md"
    actions: ActionHandler = context.application.bot_data["actions"]
    settings: Settings = context.application.bot_data["settings"]
    export_dir = settings.TEMP_DIR / "exports"

    if fmt == "json":
        export_file = actions.export_json(export_dir / "reelmind_export.json")
    else:
        export_file = actions.export_markdown(export_dir / "reelmind_export.md")

    with open(export_file, "rb") as fp:
        await update.message.reply_document(document=fp, caption=f"📦 Exported ReelMind knowledge ({fmt.upper()})")

def run_bot(pipeline: ReelPipeline, search_engine: SearchEngine, actions: ActionHandler, settings: Settings):
    pipeline.startup_recovery()
    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    
    scheduler = SchedulerService(
        db=pipeline.db,
        bot=app.bot,
        chat_id=settings.TELEGRAM_GROUP_CHAT_ID,
        pipeline=pipeline,
        canary_url=settings.CANARY_REEL_URL
    )
    scheduler.start()

    app.bot_data["pipeline"] = pipeline
    app.bot_data["search_engine"] = search_engine
    app.bot_data["actions"] = actions
    app.bot_data["settings"] = settings
    app.bot_data["scheduler"] = scheduler

    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(CommandHandler("canary", handle_canary))
    app.add_handler(CommandHandler("ask", handle_ask))
    app.add_handler(CommandHandler("grocery", handle_grocery))
    app.add_handler(CommandHandler("code", handle_code))
    app.add_handler(CommandHandler("edit", handle_edit))
    app.add_handler(CommandHandler("export", handle_export))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    logger.info("ReelMind Bot is running! Share a Reel URL, ask a question, or use inline buttons.")
    app.run_polling()

def main():
    parser = argparse.ArgumentParser(description="ReelMind — Personal AI Second Brain")
    parser.add_argument("--url", type=str, help="Process a single Instagram Reel URL immediately via CLI")
    args = parser.parse_args()

    settings = Settings()
    pipeline = ReelPipeline(settings)
    pipeline.startup_recovery()
    search_engine = SearchEngine(pipeline.db, pipeline.analyzer)
    actions = ActionHandler(pipeline.db, pipeline.analyzer)

    if args.url:
        logger.info(f"Processing URL via CLI: {args.url}")
        success, result, _ = asyncio.run(pipeline.process_url(args.url))
        print(f"[{'SUCCESS' if success else 'FAILED'}] {result}")
    else:
        run_bot(pipeline, search_engine, actions, settings)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create Dockerfile & docker-compose.yml for one-command deployment**

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  reelmind:
    build: .
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./data:/app/data
```

- [ ] **Step 3: Write comprehensive README.md with 3-minute self-hosted guide, command reference, and hackathon pitch guide**
- [ ] **Step 4: Run full automated test suite across all modules**

Run: `pytest -v`  
Expected: All tests PASS

- [ ] **Step 5: Commit changes**

```bash
git add main.py Dockerfile docker-compose.yml README.md
git commit -m "feat: master edition with user intent, state machine, truthful provenance and feedback loop"
```
