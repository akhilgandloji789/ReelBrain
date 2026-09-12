# ReelMind: Personal AI Second Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ReelMind — a self-hosted, Telegram-native multimodal personal second brain that turns saved short-form content into structured, actionable knowledge, anchored on **Memory → Evidence → Retrieval → Action**.

**Architecture:** A decoupled, modular Python 3.10+ application featuring:
- **Resilient Ingestion & Extraction:** Telegram Share Sheet link receiver with canonical shortcode deduplication + `yt-dlp` wrapped with `tenacity` exponential backoff retries.
- **Evidence-Grounded Multimodal AI:** Gemini 2.0 Flash with extensible typed schemas (Recipe, Workout, Tech, Ideas), grounded claims with evidence timestamps, video timeline, and modern `models/gemini-embedding-001` vectors (with tenacity retries).
- **Canonical Knowledge Base & Data Sovereignty:** SQLite source of truth with FTS5, vector search, synchronized re-embedding on user edits (`/edit`), and full exportability (`/export`).
- **Interactive Actions & Proactive Health:** Direct Telegram command hooks (`/ask`, `/grocery`, `/code`, `/edit`, `/export`), Sunday Action Review, and a weekly background Canary test to alert on `yt-dlp` breaks.

**Tech Stack:** Python 3.10+, `python-telegram-bot>=21.0`, `google-genai>=0.1.0`, `yt-dlp>=2024.8.6`, `pydantic>=2.7.0`, `pydantic-settings>=2.2.0`, `apscheduler>=3.10.0`, `tenacity>=8.2.0`, `numpy>=1.26.0`, `pytest`, `pytest-asyncio`, `ffmpeg`.

**Spec:** [`docs/superpowers/specs/2026-09-12-reel-ai-knowledge-agent-design.md`](file:///c:/Akhil/Instagram/docs/superpowers/specs/2026-09-12-reel-ai-knowledge-agent-design.md)

## Global Constraints
- **Zero Instagram account logins or scraping credentials** — zero account ban risk.
- **Designed for personal self-hosted use within free tiers** (Gemini free tier, Telegram Bot API, SQLite).
- **SQLite is the canonical source of truth**; Telegram is purely a presentation and interaction view.
- **Evidence-backed outputs:** All claims, ingredients, and steps must be paired with video timeline evidence timestamps.
- **Continuous index integrity:** Any edit via `/edit` must automatically re-embed the record in the same operation.
- **Data sovereignty:** `/export` must allow exporting all records to clean Markdown or JSON anytime.
- **Zero disk bloat:** Temporary MP4 video and JPEG frame files must be cleaned up immediately after dispatch.
- **Original links:** The source Reel link must always appear at the bottom of the Telegram note.

---

### Task 1: Project Setup, Dependencies & Configuration

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
  - `CANARY_REEL_URL: str | None`
  - `DATA_DIR: Path`
  - `TEMP_DIR: Path`
  - `topic_map: dict[str, int | None]`

- [ ] **Step 1: Write failing test for configuration loading**

```python
# tests/test_config.py
import pytest
from pathlib import Path
from config import Settings

def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:fake_token")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123456789")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
    
    settings = Settings()
    assert settings.TELEGRAM_BOT_TOKEN == "12345:fake_token"
    assert str(settings.TELEGRAM_GROUP_CHAT_ID) == "-100123456789"
    assert settings.GEMINI_API_KEY == "fake_gemini_key"
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
    CANARY_REEL_URL: str | None = Field(default=None, description="Stable public Reel URL for weekly health canary")
    
    # Topic Thread IDs (Optional)
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
    def topic_map(self) -> dict[str, int | None]:
        return {
            "Recipes & Food": self.TOPIC_RECIPES_THREAD_ID,
            "Tech & Coding": self.TOPIC_TECH_THREAD_ID,
            "Fitness & Health": self.TOPIC_FITNESS_THREAD_ID,
            "Finance & Investing": self.TOPIC_FINANCE_THREAD_ID,
            "Books & Ideas": self.TOPIC_BOOKS_THREAD_ID,
            "Travel": self.TOPIC_TRAVEL_THREAD_ID,
            "General & Other": self.TOPIC_GENERAL_THREAD_ID,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add requirements.txt .env.example config.py tests/test_config.py
git commit -m "feat: setup project dependencies with tenacity and configuration loader"
```

---

### Task 2: Canonical SQLite Database with FTS5, Deduplication & Export Queries

**Files:**
- Create: `modules/__init__.py`
- Create: `modules/storage.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Produces: `ReelDatabase` class:
  - `add_reel(...) -> int`
  - `is_processed(source_id: str) -> bool`
  - `get_reel_by_source_id(source_id: str) -> dict | None`
  - `get_reel_by_id(reel_id: int) -> dict | None`
  - `update_structured_data(reel_id: int, structured_data: dict, embedding: list[float] | None = None) -> bool`
  - `search_fts(query: str, limit: int = 5) -> list[dict]`
  - `get_recent_reels(days: int = 7) -> list[dict]`
  - `get_all_reels() -> list[dict]`
  - `get_all_embeddings() -> list[dict]`

- [ ] **Step 1: Write failing test for SQLite storage, FTS5, and re-embedding update**

```python
# tests/test_storage.py
import pytest
from pathlib import Path
from modules.storage import ReelDatabase

def test_database_lifecycle_and_update(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    
    shortcode = "C-xyz123"
    url = f"https://www.instagram.com/reel/{shortcode}/"
    assert not db.is_processed(shortcode)
    
    reel_id = db.add_reel(
        source_id=shortcode,
        source_url=url,
        title="10-Minute High-Protein Oats",
        content_type="Recipes & Food",
        tldr="Quick no-cook breakfast.",
        timeline=[{"timestamp": 8.0, "label": "Ingredients"}],
        claims=[{"claim": "35g protein", "evidence_timestamp": 8.0}],
        structured_data={"recipe": {"ingredients": [{"item": "Oats", "quantity": "500g"}]}},
        tags=["#Recipes", "#Nutrition"],
        thread_id=12,
        embedding=[0.1, 0.2]
    )
    
    assert reel_id > 0
    assert db.is_processed(shortcode)
    
    # Test update structured data and re-embedding
    new_structured = {"recipe": {"ingredients": [{"item": "Oats", "quantity": "250g"}]}}
    new_embedding = [0.3, 0.4]
    ok = db.update_structured_data(reel_id, new_structured, embedding=new_embedding)
    assert ok is True
    
    record = db.get_reel_by_id(reel_id)
    assert "250g" in record["structured_data_json"]
    assert "500g" not in record["structured_data_json"]
    
    # Test export query
    all_reels = db.get_all_reels()
    assert len(all_reels) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.storage')

- [ ] **Step 3: Implement modules/storage.py**

```python
# modules/storage.py
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

class ReelDatabase:
    def __init__(self, db_path: Path | str = Path("data/reelminds.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT UNIQUE NOT NULL,
                    source_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    tldr TEXT NOT NULL,
                    timeline_json TEXT NOT NULL,
                    claims_json TEXT NOT NULL,
                    structured_data_json TEXT NOT NULL,
                    tags_csv TEXT NOT NULL,
                    thread_id INTEGER,
                    embedding_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS reels_fts USING fts5(
                    id UNINDEXED,
                    title,
                    tldr,
                    tags_csv,
                    structured_data_json,
                    content=reels,
                    content_rowid=id
                )
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS reels_ai AFTER INSERT ON reels BEGIN
                    INSERT INTO reels_fts(rowid, id, title, tldr, tags_csv, structured_data_json)
                    VALUES (new.id, new.id, new.title, new.tldr, new.tags_csv, new.structured_data_json);
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS reels_ad AFTER DELETE ON reels BEGIN
                    INSERT INTO reels_fts(reels_fts, rowid, id, title, tldr, tags_csv, structured_data_json)
                    VALUES('delete', old.id, old.id, old.title, old.tldr, old.tags_csv, old.structured_data_json);
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS reels_au AFTER UPDATE ON reels BEGIN
                    INSERT INTO reels_fts(reels_fts, rowid, id, title, tldr, tags_csv, structured_data_json)
                    VALUES('delete', old.id, old.id, old.title, old.tldr, old.tags_csv, old.structured_data_json);
                    INSERT INTO reels_fts(rowid, id, title, tldr, tags_csv, structured_data_json)
                    VALUES (new.id, new.id, new.title, new.tldr, new.tags_csv, new.structured_data_json);
                END;
            """)
            conn.commit()

    def is_processed(self, source_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM reels WHERE source_id = ?", (source_id,))
            return cursor.fetchone() is not None

    def add_reel(
        self,
        source_id: str,
        source_url: str,
        title: str,
        content_type: str,
        tldr: str,
        timeline: list,
        claims: list,
        structured_data: dict,
        tags: list,
        thread_id: int | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        tags_csv = ", ".join(tags)
        embedding_str = json.dumps(embedding) if embedding else None
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reels (
                    source_id, source_url, title, content_type, tldr,
                    timeline_json, claims_json, structured_data_json,
                    tags_csv, thread_id, embedding_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    source_url,
                    title,
                    content_type,
                    tldr,
                    json.dumps(timeline),
                    json.dumps(claims),
                    json.dumps(structured_data),
                    tags_csv,
                    thread_id,
                    embedding_str,
                    now,
                    now
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_reel_by_source_id(self, source_id: str) -> dict | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM reels WHERE source_id = ?", (source_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_reel_by_id(self, reel_id: int) -> dict | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM reels WHERE id = ?", (reel_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_structured_data(self, reel_id: int, structured_data: dict, embedding: list[float] | None = None) -> bool:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            if embedding is not None:
                cursor = conn.execute(
                    "UPDATE reels SET structured_data_json = ?, embedding_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(structured_data), json.dumps(embedding), now, reel_id)
                )
            else:
                cursor = conn.execute(
                    "UPDATE reels SET structured_data_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(structured_data), now, reel_id)
                )
            conn.commit()
            return cursor.rowcount > 0

    def search_fts(self, query: str, limit: int = 5) -> list[dict]:
        clean_query = query.replace('"', '""').strip()
        if not clean_query:
            return []
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.* FROM reels r
                JOIN reels_fts ON r.id = reels_fts.id
                WHERE reels_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (clean_query, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_reels(self, days: int = 7) -> list[dict]:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM reels WHERE created_at >= ? ORDER BY created_at DESC", (since,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_reels(self) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM reels ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_all_embeddings(self) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id, title, content_type, tldr, source_url, embedding_json FROM reels WHERE embedding_json IS NOT NULL")
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["embedding"] = json.loads(d["embedding_json"])
                results.append(d)
            return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/__init__.py modules/storage.py tests/test_storage.py
git commit -m "feat: implement SQLite canonical store with atomic re-embedding and export queries"
```

---

### Task 3: Downloader Adapter with Tenacity Retries & Failure Isolation

**Files:**
- Create: `modules/downloader.py`
- Create: `tests/test_downloader.py`

**Interfaces:**
- Produces: `Downloader` class:
  - `parse_source_id(raw_text: str) -> tuple[str | None, str | None]`
  - `download_video(url: str, output_dir: Path) -> Path` (Retries 3 times with exponential backoff on transient errors)

- [ ] **Step 1: Write failing test for downloader and retry behavior**

```python
# tests/test_downloader.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.downloader import Downloader, DownloadError

def test_parse_source_id():
    downloader = Downloader()
    sid, canonical = downloader.parse_source_id("https://instagram.com/reel/C-12345Abc/?igsh=test")
    assert sid == "C-12345Abc"
    assert canonical == "https://www.instagram.com/reel/C-12345Abc/"

@patch("modules.downloader.YoutubeDL")
def test_download_video_success(mock_ydl_class, tmp_path):
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.return_value = {"id": "C-12345Abc", "ext": "mp4"}
    
    video_file = tmp_path / "C-12345Abc.mp4"
    video_file.write_text("dummy video")
    
    downloader = Downloader()
    result = downloader.download_video("https://instagram.com/reel/C-12345Abc/", output_dir=tmp_path)
    assert result.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_downloader.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.downloader')

- [ ] **Step 3: Implement modules/downloader.py with tenacity**

```python
# modules/downloader.py
import re
from pathlib import Path
from yt_dlp import YoutubeDL
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

REEL_REGEX = re.compile(r'https?://(?:www\.)?instagram\.com/(?:reel|reels|p)/([a-zA-Z0-9_-]+)')

class DownloadError(Exception):
    pass

class Downloader:
    def parse_source_id(self, raw_text: str) -> tuple[str | None, str | None]:
        match = REEL_REGEX.search(raw_text)
        if not match:
            return None, None
        shortcode = match.group(1)
        is_post = "/p/" in match.group(0)
        canonical_type = "p" if is_post else "reel"
        canonical_url = f"https://www.instagram.com/{canonical_type}/{shortcode}/"
        return shortcode, canonical_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type(DownloadError),
        reraise=True
    )
    def download_video(self, url: str, output_dir: Path) -> Path:
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
                    return expected_mp4
                
                for f in output_dir.glob(f"{video_id}.*"):
                    return f
                    
                raise DownloadError(f"Video file was not found after download for {url}")
        except Exception as e:
            raise DownloadError(f"Download error on {url}: {str(e)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_downloader.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/downloader.py tests/test_downloader.py
git commit -m "feat: implement downloader adapter with tenacity exponential backoff"
```

---

### Task 4: Multimodal Analyzer with Typed Schemas, Evidence Timestamps & Gemini Embedding 001

**Files:**
- Create: `modules/analyzer.py`
- Create: `tests/test_analyzer.py`

**Interfaces:**
- Produces:
  - `ReelKnowledgeObject` (Pydantic Model with timeline & claims with evidence)
  - `ReelAnalyzer` class:
    - `analyze_video(video_path: Path) -> ReelKnowledgeObject`
    - `generate_embedding(text: str) -> list[float]` (Using `models/gemini-embedding-001`)

- [ ] **Step 1: Write failing test for analyzer schemas and retry configuration**

```python
# tests/test_analyzer.py
import pytest
from pathlib import Path
from modules.analyzer import ReelKnowledgeObject, ReelAnalyzer

def test_knowledge_object_evidence():
    data = {
        "title": "Tuscan Garlic Chicken",
        "content_type": "Recipes & Food",
        "tldr": "20-minute dinner.",
        "timeline": [{"timestamp": 0.0, "label": "Intro"}],
        "claims_with_evidence": [{"claim": "Bake at 180C", "evidence_timestamp": 12.0, "confidence": "high"}],
        "key_takeaways": ["Sear chicken well"],
        "highlight_timestamps": [12.0],
        "structured_data": {"recipe": {"ingredients": [{"item": "Chicken", "quantity": "500g"}]}},
        "tags": ["#Recipes"]
    }
    obj = ReelKnowledgeObject.model_validate(data)
    assert obj.claims_with_evidence[0].evidence_timestamp == 12.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzer.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.analyzer')

- [ ] **Step 3: Implement modules/analyzer.py using gemini-embedding-001 and tenacity retries**

```python
# modules/analyzer.py
import json
import time
from pathlib import Path
from typing import Literal, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

ContentType = Literal[
    "Recipes & Food",
    "Tech & Coding",
    "Fitness & Health",
    "Finance & Investing",
    "Books & Ideas",
    "Travel",
    "General & Other"
]

class TimelineItem(BaseModel):
    timestamp: float = Field(description="Time in seconds (e.g. 0.0, 12.5)")
    label: str = Field(description="Stage name, e.g. 'Hook', 'Ingredients', 'Core Technique', 'Finished Result'")

class ClaimWithEvidence(BaseModel):
    claim: str = Field(description="Specific fact, measurement, rule or instruction asserted")
    evidence_timestamp: float = Field(description="Exact second in the video where this claim is spoken or visually shown")
    confidence: Literal["high", "medium"] = Field(default="high")

class ReelKnowledgeObject(BaseModel):
    title: str = Field(description="Descriptive, engaging, searchable title")
    content_type: ContentType = Field(description="Best fitting domain category")
    tldr: str = Field(description="Crisp 1-2 sentence executive summary")
    timeline: list[TimelineItem] = Field(description="Key narrative progression stages of the video")
    claims_with_evidence: list[ClaimWithEvidence] = Field(description="Specific claims linked to exact timestamps")
    key_takeaways: list[str] = Field(description="Top 2-4 bulleted takeaways")
    highlight_timestamps: list[float] = Field(description="2 to 4 timestamps for high-res photo snapshot extraction")
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible typed payload (e.g. recipe: {ingredients, instructions}, tech: {code_snippets, tools}, workout: {exercises})"
    )
    tags: list[str] = Field(description="3-5 relevant hashtags with #")

ANALYSIS_SYSTEM_INSTRUCTION = """
You are ReelMind, an expert multimodal knowledge extraction agent.
Analyze the provided Reel (both visual video frames and spoken audio).
Your job is not to passively summarize, but to index and extract structured, verifiable knowledge:
1. Classify the content type accurately.
2. Build a video Timeline with key moments.
3. Extract specific Claims with Evidence timestamps (link every fact/measurement to its exact second).
4. Extract 2-4 highlight timestamps where the most informative visual detail appears (e.g. ingredients list, final result, code screen, technique).
5. Build the structured_data payload:
   - For Recipes: ingredients with quantities, instructions, prep time.
   - For Tech/Coding: tools/languages, clean code snippets/commands.
   - For Workout: target muscles, exercises with sets and reps.
   - For Books/Ideas: key concepts, quotes, action items.
Always output strict JSON conforming to the schema.
"""

class ReelAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
    def analyze_video(self, video_path: Path) -> ReelKnowledgeObject:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        uploaded_file = self.client.files.upload(file=video_path)
        while uploaded_file.state == "PROCESSING":
            time.sleep(2)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    uploaded_file,
                    "Index and extract structured knowledge from this video and audio with evidence timestamps."
                ],
                config=types.GenerateContentConfig(
                    system_instruction=ANALYSIS_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=ReelKnowledgeObject,
                    temperature=0.2,
                )
            )
            data = json.loads(response.text)
            return ReelKnowledgeObject.model_validate(data)
        finally:
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
    def generate_embedding(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text
        )
        return response.embedding.values
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/analyzer.py tests/test_analyzer.py
git commit -m "feat: implement analyzer with gemini-embedding-001 and tenacity retries"
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

- [ ] **Step 1: Write failing test for frame extractor**

```python
# tests/test_frame_extractor.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.frame_extractor import FrameExtractor

@patch("subprocess.run")
def test_extract_frames_at_evidence_timestamps(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    video = tmp_path / "vid.mp4"
    video.write_text("dummy")
    
    extractor = FrameExtractor()
    frames = extractor.extract_frames(video, [8.5, 22.0], tmp_path / "frames")
    
    assert len(frames) == 2
    assert mock_run.call_count == 2
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
            output_frame = output_dir / f"{video_stem}_evidence_{idx}_{int(ts)}s.jpg"
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

### Task 6: Telegram Publisher with Forum Topic Routing & Evidence Formatting

**Files:**
- Create: `modules/publisher.py`
- Create: `tests/test_publisher.py`

**Interfaces:**
- Produces: `TelegramPublisher` class:
  - `format_evidence_note_html(knowledge: ReelKnowledgeObject, original_url: str) -> str`
  - `publish_reel(knowledge: ReelKnowledgeObject, frame_paths: list[Path], original_url: str) -> int | None`

- [ ] **Step 1: Write failing test for HTML evidence formatting and topic routing**

```python
# tests/test_publisher.py
import pytest
from pathlib import Path
from modules.analyzer import ReelKnowledgeObject, TimelineItem, ClaimWithEvidence
from modules.publisher import TelegramPublisher

@pytest.fixture
def sample_knowledge():
    return ReelKnowledgeObject(
        title="10-Minute High-Protein Oats",
        content_type="Recipes & Food",
        tldr="Quick no-cook meal prep breakfast.",
        timeline=[
            TimelineItem(timestamp=8.0, label="Ingredients breakdown"),
            TimelineItem(timestamp=45.0, label="Finished jar")
        ],
        claims_with_evidence=[
            ClaimWithEvidence(claim="Base: 50g oats, 1 scoop whey", evidence_timestamp=8.0),
            ClaimWithEvidence(claim="Lasts 4 days refrigerated", evidence_timestamp=50.0)
        ],
        key_takeaways=["No cooking needed", "35g protein"],
        highlight_timestamps=[8.0, 45.0],
        structured_data={"recipe": {"ingredients": [{"item": "Oats", "quantity": "50g"}]}},
        tags=["#Recipes", "#Nutrition"]
    )

def test_format_evidence_note_html_structure(sample_knowledge):
    publisher = TelegramPublisher(bot_token="fake", group_chat_id="-100")
    url = "https://www.instagram.com/reel/C-test123/"
    
    html = publisher.format_evidence_note_html(sample_knowledge, original_url=url)
    assert "🎬 <b>10-Minute High-Protein Oats</b>" in html
    assert "⏱️ <b>Timeline:</b>" in html
    assert "• 00:08 — Ingredients breakdown" in html
    assert "⚡ <b>Key Highlights & Evidence:</b>" in html
    assert "<i>(00:08)</i>" in html
    assert html.strip().endswith(f'<a href="{url}">{url}</a>')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_publisher.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.publisher')

- [ ] **Step 3: Implement modules/publisher.py**

```python
# modules/publisher.py
import html
from pathlib import Path
from telegram import Bot, InputMediaPhoto
from modules.analyzer import ReelKnowledgeObject

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
        return self.topic_map.get(category)

    def format_evidence_note_html(self, knowledge: ReelKnowledgeObject, original_url: str) -> str:
        safe_title = html.escape(knowledge.title)
        safe_tldr = html.escape(knowledge.tldr)
        
        timeline_lines = [
            f"• {format_timestamp(item.timestamp)} — {html.escape(item.label)}"
            for item in knowledge.timeline
        ]
        timeline_block = "\n".join(timeline_lines)

        evidence_lines = [
            f"• {html.escape(c.claim)} <i>({format_timestamp(c.evidence_timestamp)})</i>"
            for c in knowledge.claims_with_evidence
        ]
        if not evidence_lines:
            evidence_lines = [f"• {html.escape(t)}" for t in knowledge.key_takeaways]
        evidence_block = "\n".join(evidence_lines)

        tags_line = " ".join(knowledge.tags)
        safe_url = html.escape(original_url)

        return (
            f"🎬 <b>{safe_title}</b>\n\n"
            f"📌 <b>TL;DR:</b>\n{safe_tldr}\n\n"
            f"⏱️ <b>Timeline:</b>\n{timeline_block}\n\n"
            f"⚡ <b>Key Highlights & Evidence:</b>\n{evidence_block}\n\n"
            f"🏷️ <i>{tags_line}</i>\n\n"
            f'🔗 <b>Original Reel:</b> <a href="{safe_url}">{safe_url}</a>'
        )

    async def publish_reel(self, knowledge: ReelKnowledgeObject, frame_paths: list[Path], original_url: str) -> int | None:
        thread_id = self.get_thread_id(knowledge.content_type)
        
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

        text_content = self.format_evidence_note_html(knowledge, original_url)
        await self.bot.send_message(
            chat_id=self.group_chat_id,
            message_thread_id=thread_id,
            text=text_content,
            parse_mode="HTML",
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
git commit -m "feat: implement telegram publisher with evidence timeline and HTML formatting"
```

---

### Task 7: End-to-End Pipeline & CLI Runner (v1.0 MVP)

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `ReelPipeline` class:
  - `process_url(raw_input: str) -> tuple[bool, str, int | None]`

- [ ] **Step 1: Write failing test for complete pipeline execution**

```python
# tests/test_pipeline.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from pipeline import ReelPipeline
from modules.analyzer import ReelKnowledgeObject

@pytest.mark.asyncio
async def test_pipeline_deduplication_and_execution(tmp_path):
    settings = MagicMock()
    settings.TEMP_DIR = tmp_path
    
    mock_db = MagicMock()
    mock_db.is_processed.return_value = False
    
    mock_downloader = MagicMock()
    video_file = tmp_path / "test.mp4"
    video_file.write_text("dummy")
    mock_downloader.parse_source_id.return_value = ("C-test1", "https://instagram.com/reel/C-test1/")
    mock_downloader.download_video.return_value = video_file
    
    mock_analyzer = MagicMock()
    knowledge = ReelKnowledgeObject(
        title="Test Reel",
        content_type="Tech & Coding",
        tldr="Summary",
        timeline=[],
        claims_with_evidence=[],
        key_takeaways=["Point 1"],
        highlight_timestamps=[1.0],
        tags=["#tech"]
    )
    mock_analyzer.analyze_video.return_value = knowledge
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
    
    success, msg, thread_id = await pipeline.process_url("https://instagram.com/reel/C-test1/")
    assert success is True
    assert "Test Reel" in msg
    mock_db.add_reel.assert_called_once()
    assert not video_file.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'pipeline')

- [ ] **Step 3: Implement pipeline.py**

```python
# pipeline.py
import asyncio
from pathlib import Path
from config import Settings
from modules.storage import ReelDatabase
from modules.downloader import Downloader
from modules.analyzer import ReelAnalyzer
from modules.frame_extractor import FrameExtractor
from modules.publisher import TelegramPublisher

class ReelPipeline:
    def __init__(
        self,
        settings: Settings,
        db: ReelDatabase | None = None,
        downloader: Downloader | None = None,
        analyzer: ReelAnalyzer | None = None,
        extractor: FrameExtractor | None = None,
        publisher: TelegramPublisher | None = None,
    ):
        self.settings = settings
        self.db = db or ReelDatabase(settings.DATA_DIR / "reelminds.db")
        self.downloader = downloader or Downloader()
        self.analyzer = analyzer or ReelAnalyzer(api_key=settings.GEMINI_API_KEY)
        self.extractor = extractor or FrameExtractor()
        self.publisher = publisher or TelegramPublisher(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            group_chat_id=settings.TELEGRAM_GROUP_CHAT_ID,
            topic_map=settings.topic_map
        )
        self._lock = asyncio.Lock()

    async def process_url(self, raw_input: str) -> tuple[bool, str, int | None]:
        async with self._lock:
            source_id, canonical_url = self.downloader.parse_source_id(raw_input)
            if not source_id or not canonical_url:
                return False, "Not a valid Instagram Reel or Post URL.", None

            # Deduplication
            if self.db.is_processed(source_id):
                existing = self.db.get_reel_by_source_id(source_id)
                category = existing.get("content_type", "General") if existing else "General"
                title = existing.get("title", "Reel") if existing else "Reel"
                return True, f"⚠️ Already saved under <b>{category}</b>:\n📌 <i>{title}</i>", existing.get("thread_id")

            video_dir = self.settings.TEMP_DIR / "videos"
            frames_dir = self.settings.TEMP_DIR / "frames"
            video_file: Path | None = None
            frame_paths: list[Path] = []

            try:
                # 1. Download with retry
                video_file = self.downloader.download_video(canonical_url, output_dir=video_dir)

                # 2. Multimodal AI extraction
                knowledge = self.analyzer.analyze_video(video_file)

                # 3. Extract evidence keyframe snapshots
                frame_paths = self.extractor.extract_frames(
                    video_path=video_file,
                    timestamps=knowledge.highlight_timestamps,
                    output_dir=frames_dir
                )

                # 4. Generate text embedding for semantic search
                embedding: list[float] | None = None
                try:
                    embed_text = f"{knowledge.title}\n{knowledge.tldr}\n{' '.join(knowledge.key_takeaways)}"
                    embedding = self.analyzer.generate_embedding(embed_text)
                except Exception:
                    pass

                # 5. Publish to Telegram
                thread_id = await self.publisher.publish_reel(
                    knowledge=knowledge,
                    frame_paths=frame_paths,
                    original_url=canonical_url
                )

                # 6. Store in canonical SQLite database
                self.db.add_reel(
                    source_id=source_id,
                    source_url=canonical_url,
                    title=knowledge.title,
                    content_type=knowledge.content_type,
                    tldr=knowledge.tldr,
                    timeline=[t.model_dump() for t in knowledge.timeline],
                    claims=[c.model_dump() for c in knowledge.claims_with_evidence],
                    structured_data=knowledge.structured_data,
                    tags=knowledge.tags,
                    thread_id=thread_id,
                    embedding=embedding
                )

                return True, f"✅ Indexed <b>{knowledge.title}</b> into <i>{knowledge.content_type}</i>", thread_id

            except Exception as e:
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
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: implement pipeline orchestrator with deduplication and auto-cleanup"
```

---

### Task 8: Hybrid Search Engine & Conversational `/ask` (v1.1)

**Files:**
- Create: `modules/search.py`
- Create: `tests/test_search.py`

**Interfaces:**
- Produces: `SearchEngine` class:
  - `hybrid_search(query: str, top_k: int = 5) -> list[dict]`
  - `answer_conversational_query(query: str) -> str`

- [ ] **Step 1: Write failing test for hybrid search engine**

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
    
    mock_db.search_fts.return_value = [
        {"id": 1, "title": "Mobility Routine", "content_type": "Fitness & Health", "tldr": "Warmup", "source_url": "url1"}
    ]
    mock_db.get_all_embeddings.return_value = [
        {"id": 1, "title": "Mobility Routine", "content_type": "Fitness & Health", "tldr": "Warmup", "source_url": "url1", "embedding": [0.9, 0.1]}
    ]
    
    engine = SearchEngine(db=mock_db, analyzer=mock_analyzer)
    results = engine.hybrid_search("pre gym warmup")
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
            context_lines.append(
                f"[{idx}] Title: {m['title']}\n"
                f"Category: {m['content_type']}\n"
                f"Summary: {m['tldr']}\n"
                f"Source: {m['source_url']}"
            )
        context_str = "\n\n".join(context_lines)

        prompt = f"""
        You are ReelMind personal AI assistant. A user is asking a question about their saved reels collection:
        "{query}"

        Here is the relevant saved knowledge from their collection:
        {context_str}

        Answer the user's question directly, clearly, and concisely in Telegram HTML format. Cite the specific reel title and include its source link.
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
git commit -m "feat: implement hybrid search engine (FTS5 + vector similarity) and /ask conversational retriever"
```

---

### Task 9: Action Hooks (`/grocery`, `/code`), Synchronized Re-Embedding (`/edit`), and Data Sovereignty (`/export`) (v1.2)

**Files:**
- Create: `modules/actions.py`
- Create: `tests/test_actions.py`

**Interfaces:**
- Produces: `ActionHandler` class:
  - `generate_grocery_list(reel: dict) -> str`
  - `generate_code_block(reel: dict) -> str`
  - `apply_edit(reel_id: int, old_str: str, new_str: str) -> tuple[bool, str]` (Atomically re-embeds!)
  - `export_markdown(output_path: Path) -> Path`
  - `export_json(output_path: Path) -> Path`

- [ ] **Step 1: Write failing test for action generation, synchronized re-embedding, and export**

```python
# tests/test_actions.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from modules.actions import ActionHandler

def test_grocery_and_code_generation():
    handler = ActionHandler(db=MagicMock(), analyzer=MagicMock())
    reel_recipe = {
        "title": "Protein Pancakes",
        "structured_data_json": '{"recipe": {"ingredients": [{"item": "Oats", "quantity": "50g"}]}}'
    }
    grocery = handler.generate_grocery_list(reel_recipe)
    assert "🛒 <b>Grocery List: Protein Pancakes</b>" in grocery
    assert "• [ ] 50g Oats" in grocery

def test_apply_edit_triggers_reembedding():
    mock_db = MagicMock()
    mock_analyzer = MagicMock()
    mock_analyzer.generate_embedding.return_value = [0.99, 0.01]
    
    mock_db.get_reel_by_id.return_value = {
        "id": 1,
        "title": "Oats Recipe",
        "tldr": "Breakfast",
        "structured_data_json": '{"recipe": {"ingredients": [{"item": "Oats", "quantity": "500g"}]}}'
    }
    mock_db.update_structured_data.return_value = True
    
    handler = ActionHandler(db=mock_db, analyzer=mock_analyzer)
    success, msg = handler.apply_edit(reel_id=1, old_str="500g", new_str="250g")
    
    assert success is True
    # Re-embedding was called
    mock_analyzer.generate_embedding.assert_called_once()
    # update_structured_data was called with new embedding
    mock_db.update_structured_data.assert_called_once()
    call_args = mock_db.update_structured_data.call_args
    assert call_args[1]["embedding"] == [0.99, 0.01]

def test_export_markdown_and_json(tmp_path):
    mock_db = MagicMock()
    mock_db.get_all_reels.return_value = [
        {"id": 1, "title": "Oats Recipe", "content_type": "Recipes & Food", "tldr": "Summary", "source_url": "url1", "tags_csv": "#recipe"}
    ]
    handler = ActionHandler(db=mock_db, analyzer=MagicMock())
    
    md_file = handler.export_markdown(tmp_path / "export.md")
    assert md_file.exists()
    assert "Oats Recipe" in md_file.read_text(encoding="utf-8")
    
    json_file = handler.export_json(tmp_path / "export.json")
    assert json_file.exists()
    assert "Oats Recipe" in json_file.read_text(encoding="utf-8")
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

    def generate_grocery_list(self, reel: dict) -> str:
        title = html.escape(reel.get("title", "Recipe"))
        structured = json.loads(reel.get("structured_data_json", "{}"))
        recipe = structured.get("recipe", {})
        ingredients = recipe.get("ingredients", [])
        
        if not ingredients:
            return f"⚠️ No structured ingredients found for <b>{title}</b>."
            
        lines = [f"🛒 <b>Grocery List: {title}</b>\n"]
        for ing in ingredients:
            item = html.escape(ing.get("item", ""))
            qty = html.escape(ing.get("quantity", ""))
            line = f"• [ ] {qty} {item}".strip()
            lines.append(line)
            
        return "\n".join(lines)

    def generate_code_block(self, reel: dict) -> str:
        title = html.escape(reel.get("title", "Tech Tutorial"))
        structured = json.loads(reel.get("structured_data_json", "{}"))
        tech = structured.get("tech", {})
        snippets = tech.get("code_snippets", [])
        
        if not snippets:
            return f"⚠️ No code snippets found for <b>{title}</b>."
            
        lines = [f"💻 <b>Code & Commands: {title}</b>\n"]
        for s in snippets:
            lines.append(f"<code>{html.escape(s)}</code>\n")
            
        return "\n".join(lines)

    def apply_edit(self, reel_id: int, old_str: str, new_str: str) -> tuple[bool, str]:
        reel = self.db.get_reel_by_id(reel_id)
        if not reel:
            return False, "Reel not found."
            
        raw_json = reel["structured_data_json"]
        if old_str not in raw_json:
            return False, f"Could not find '{old_str}' in reel structured data."
            
        updated_json_str = raw_json.replace(old_str, new_str)
        updated_dict = json.loads(updated_json_str)

        # Synchronously regenerate vector embedding to keep search index accurate
        new_embedding: list[float] | None = None
        try:
            embed_text = f"{reel['title']}\n{reel['tldr']}\n{updated_json_str}"
            new_embedding = self.analyzer.generate_embedding(embed_text)
        except Exception:
            pass

        success = self.db.update_structured_data(reel_id, updated_dict, embedding=new_embedding)
        return success, f"Updated '{old_str}' → '{new_str}' and refreshed search index."

    def export_markdown(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reels = self.db.get_all_reels()

        lines = ["# ReelMind Knowledge Export\n"]
        for r in reels:
            lines.append(f"## {r['title']}")
            lines.append(f"- **Category:** {r['content_type']}")
            lines.append(f"- **Summary:** {r['tldr']}")
            lines.append(f"- **Original Source:** {r['source_url']}")
            lines.append(f"- **Tags:** {r['tags_csv']}")
            lines.append("\n```json")
            lines.append(r["structured_data_json"])
            lines.append("```\n---\n")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def export_json(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reels = self.db.get_all_reels()
        output_path.write_text(json.dumps(reels, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_actions.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/actions.py tests/test_actions.py
git commit -m "feat: implement actions (/grocery, /code), synchronized re-embedding (/edit) and data export (/export)"
```

---

### Task 10: Sunday Action Review & Proactive Canary Health Check (v1.3)

**Files:**
- Create: `modules/scheduler.py`
- Create: `tests/test_scheduler.py`

**Interfaces:**
- Produces: `SchedulerService` class:
  - `build_weekly_digest() -> str | None`
  - `run_canary_test() -> tuple[bool, str]`
  - `start()`

- [ ] **Step 1: Write failing test for weekly digest and canary test**

```python
# tests/test_scheduler.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from modules.scheduler import SchedulerService

@pytest.mark.asyncio
async def test_digest_and_canary():
    mock_db = MagicMock()
    mock_db.get_recent_reels.return_value = [
        {"title": "Oats", "content_type": "Recipes & Food", "source_url": "url1"}
    ]
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
    assert digest is not None
    assert "🍳 1 recipe saved" in digest
    
    ok, msg = await service.run_canary_test()
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

        recipes = [r for r in recent if r["content_type"] == "Recipes & Food"]
        workouts = [r for r in recent if r["content_type"] == "Fitness & Health"]
        tech = [r for r in recent if r["content_type"] == "Tech & Coding"]
        others = [r for r in recent if r not in recipes + workouts + tech]

        lines = ["🧠 <b>YOUR REELMIND — SUNDAY REVIEW</b>\n"]
        if recipes:
            lines.append(f"🍳 <b>{len(recipes)} recipe{'s' if len(recipes) > 1 else ''} saved:</b>")
            for r in recipes[:3]:
                lines.append(f"• <a href='{r['source_url']}'>{r['title']}</a>")
            lines.append("")
        if workouts:
            lines.append(f"🏋️ <b>{len(workouts)} workout{'s' if len(workouts) > 1 else ''} saved:</b>")
            for w in workouts[:3]:
                lines.append(f"• <a href='{w['source_url']}'>{w['title']}</a>")
            lines.append("")
        if tech:
            lines.append(f"💻 <b>{len(tech)} tech idea{'s' if len(tech) > 1 else ''} saved:</b>")
            for t in tech[:3]:
                lines.append(f"• <a href='{t['source_url']}'>{t['title']}</a>")
            lines.append("")
        if others:
            lines.append(f"💡 <b>{len(others)} other discovery/learning items saved.</b>\n")

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
                alert_text = f"🚨 <b>Canary Alert:</b> Pipeline failed on stable reel!\nDetails: {msg}"
                await self.bot.send_message(chat_id=self.chat_id, text=alert_text, parse_mode="HTML")
                return False, msg
            return True, "Canary health check passed."
        except Exception as e:
            alert_text = f"🚨 <b>Canary Alert:</b> Exception during pipeline health check:\n{str(e)}"
            await self.bot.send_message(chat_id=self.chat_id, text=alert_text, parse_mode="HTML")
            return False, str(e)

    def start(self):
        # Sunday Action Review at 09:00 AM
        self.scheduler.add_job(self.send_weekly_digest, "cron", day_of_week="sun", hour=9, minute=0)
        # Tuesday Canary Health Check at 03:00 AM
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
git commit -m "feat: implement Sunday review digest and proactive canary test scheduler"
```

---

### Task 11: Telegram Runner Daemon, Interactive Commands & Full Test Suite

**Files:**
- Create: `main.py`
- Create: `README.md`

- [ ] **Step 1: Implement main.py connecting all commands, exports, actions, and CLI mode**

```python
# main.py
import argparse
import asyncio
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
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
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    search_engine: SearchEngine = context.application.bot_data["search_engine"]

    if "instagram.com" in text:
        status_msg = await update.message.reply_text("⏳ Processing reel... Analyzing video & audio with Gemini...")
        success, result_text, thread_id = await pipeline.process_url(text)
        await status_msg.edit_text(result_text, parse_mode="HTML")
    else:
        answer = search_engine.answer_conversational_query(text)
        await update.message.reply_text(answer, parse_mode="HTML")

async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: <code>/ask what was that recipe with oats?</code>", parse_mode="HTML")
        return
    query = " ".join(context.args)
    search_engine: SearchEngine = context.application.bot_data["search_engine"]
    answer = search_engine.answer_conversational_query(query)
    await update.message.reply_text(answer, parse_mode="HTML")

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
    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.bot_data["pipeline"] = pipeline
    app.bot_data["search_engine"] = search_engine
    app.bot_data["actions"] = actions
    app.bot_data["settings"] = settings

    app.add_handler(CommandHandler("ask", handle_ask))
    app.add_handler(CommandHandler("export", handle_export))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    scheduler = SchedulerService(
        db=pipeline.db,
        bot=app.bot,
        chat_id=settings.TELEGRAM_GROUP_CHAT_ID,
        pipeline=pipeline,
        canary_url=settings.CANARY_REEL_URL
    )
    scheduler.start()

    logger.info("ReelMind Bot is running! Share an Instagram reel, ask a question, or type /export.")
    app.run_polling()

def main():
    parser = argparse.ArgumentParser(description="ReelMind — Personal AI Second Brain")
    parser.add_argument("--url", type=str, help="Process a single Instagram Reel URL immediately via CLI")
    args = parser.parse_args()

    settings = Settings()
    pipeline = ReelPipeline(settings)
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

- [ ] **Step 2: Write README.md with 3-minute self-hosted setup, command reference & pitch points**
- [ ] **Step 3: Run full automated test suite across all modules**

Run: `pytest -v`  
Expected: All tests PASS

- [ ] **Step 4: Commit changes**

```bash
git add main.py README.md
git commit -m "feat: complete Telegram runner, interactive commands, export handler and documentation"
```
