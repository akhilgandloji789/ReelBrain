# Instagram Reel AI Knowledge Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated, 100% free AI agent that allows users to share any Instagram Reel directly to a Telegram bot, automatically downloads the video, analyzes visual frames and spoken audio using Google Gemini 2.0 Flash, extracts crisp highlight photos with ffmpeg, and publishes organized summaries into categorized Telegram Forum Topics with the original reel link at the bottom and auto-cleanup of video files.

**Architecture:** A modular Python 3.10+ application utilizing `python-telegram-bot` for async message ingestion and forum topic publishing, `yt-dlp` for video downloading, `google-genai` SDK for multimodal video+audio analysis with structured JSON schemas, `ffmpeg` for keyframe snapshot extraction, and SQLite for persistent duplicate prevention and reel history.

**Tech Stack:** Python 3.10+, `python-telegram-bot>=21.0`, `google-genai>=0.1.0`, `yt-dlp>=2024.0.0`, `pydantic>=2.0`, `pytest`, `pytest-asyncio`, `ffmpeg`.

**Spec:** [`docs/superpowers/specs/2026-09-12-reel-ai-knowledge-agent-design.md`](file:///c:/Akhil/Instagram/docs/superpowers/specs/2026-09-12-reel-ai-knowledge-agent-design.md)

## Global Constraints
- Zero Instagram account credentials or logins (0% ban risk).
- Free tier only: Gemini 2.0 Flash free tier (1,500 requests/day) and Telegram Bot API ($0).
- Automatic cleanup: All temporary MP4 videos and extracted JPEG frames in `temp/` must be deleted after Telegram dispatch.
- Message layout: Every Telegram post must include the original Instagram Reel link at the bottom.
- Cross-platform: Must run reliably on Windows 11 PowerShell and Linux environments.

---

### Task 1: Project Setup, Dependencies & Configuration

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `config.Settings` class with fields:
  - `TELEGRAM_BOT_TOKEN: str`
  - `TELEGRAM_GROUP_CHAT_ID: int | str`
  - `GEMINI_API_KEY: str`
  - `TOPIC_MAP: dict[str, int]` (Category name -> thread_id)
  - `DATA_DIR: Path`
  - `TEMP_DIR: Path`

- [ ] **Step 1: Write the failing test for configuration loading**

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

- [ ] **Step 3: Create requirements.txt and config.py implementation**

Create `requirements.txt`:
```text
python-telegram-bot>=21.0
google-genai>=0.1.0
yt-dlp>=2024.8.6
pydantic>=2.7.0
pydantic-settings>=2.2.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

Create `.env.example`:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_from_botfather
TELEGRAM_GROUP_CHAT_ID=-1001234567890
GEMINI_API_KEY=your_gemini_api_key_from_google_ai_studio
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
    
    # Optional mapping of Category Name -> Telegram message_thread_id
    TOPIC_RECIPES_THREAD_ID: int | None = None
    TOPIC_TECH_THREAD_ID: int | None = None
    TOPIC_FITNESS_THREAD_ID: int | None = None
    TOPIC_FINANCE_THREAD_ID: int | None = None
    TOPIC_BOOKS_THREAD_ID: int | None = None
    TOPIC_GENERAL_THREAD_ID: int | None = None

    DATA_DIR: Path = Path("data")
    TEMP_DIR: Path = Path("temp")

    def model_post_init(self, __context):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        (self.TEMP_DIR / "videos").mkdir(parents=True, exist_ok=True)
        (self.TEMP_DIR / "frames").mkdir(parents=True, exist_ok=True)

    @property
    def topic_map(self) -> dict[str, int | None]:
        return {
            "Recipes & Food": self.TOPIC_RECIPES_THREAD_ID,
            "Tech & Coding": self.TOPIC_TECH_THREAD_ID,
            "Fitness & Health": self.TOPIC_FITNESS_THREAD_ID,
            "Finance & Investing": self.TOPIC_FINANCE_THREAD_ID,
            "Books & Learning": self.TOPIC_BOOKS_THREAD_ID,
            "General & Other": self.TOPIC_GENERAL_THREAD_ID,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add requirements.txt .env.example config.py tests/test_config.py
git commit -m "feat: add project configuration and dependencies"
```

---

### Task 2: SQLite Storage & Duplicate Prevention

**Files:**
- Create: `modules/__init__.py`
- Create: `modules/storage.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Produces: `ReelDatabase` class:
  - `add_reel(reel_url: str, title: str, category: str, thread_id: int | None) -> None`
  - `is_processed(reel_url: str) -> bool`
  - `get_reel(reel_url: str) -> dict | None`

- [ ] **Step 1: Write the failing test for SQLite storage**

```python
# tests/test_storage.py
import pytest
from pathlib import Path
from modules.storage import ReelDatabase

def test_database_lifecycle(tmp_path: Path):
    db_path = tmp_path / "test_reels.db"
    db = ReelDatabase(db_path)
    
    url = "https://www.instagram.com/reel/Cxyz123/"
    assert not db.is_processed(url)
    
    db.add_reel(url, title="Quick Oats Recipe", category="Recipes & Food", thread_id=12)
    assert db.is_processed(url)
    
    record = db.get_reel(url)
    assert record is not None
    assert record["title"] == "Quick Oats Recipe"
    assert record["category"] == "Recipes & Food"
    assert record["thread_id"] == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.storage')

- [ ] **Step 3: Implement modules/storage.py**

```python
# modules/storage.py
import sqlite3
from pathlib import Path
from datetime import datetime

class ReelDatabase:
    def __init__(self, db_path: Path | str = Path("data/reels.db")):
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
                CREATE TABLE IF NOT EXISTS processed_reels (
                    reel_url TEXT PRIMARY KEY,
                    title TEXT,
                    category TEXT,
                    thread_id INTEGER,
                    processed_at TEXT
                )
            """)
            conn.commit()

    def is_processed(self, reel_url: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM processed_reels WHERE reel_url = ?", (reel_url,))
            return cursor.fetchone() is not None

    def add_reel(self, reel_url: str, title: str, category: str, thread_id: int | None = None) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_reels (reel_url, title, category, thread_id, processed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (reel_url, title, category, thread_id, datetime.utcnow().isoformat())
            )
            conn.commit()

    def get_reel(self, reel_url: str) -> dict | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM processed_reels WHERE reel_url = ?", (reel_url,))
            row = cursor.fetchone()
            return dict(row) if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/__init__.py modules/storage.py tests/test_storage.py
git commit -m "feat: add SQLite storage module for processed reels"
```

---

### Task 3: Reel Video Downloader Module (`yt-dlp`)

**Files:**
- Create: `modules/downloader.py`
- Create: `tests/test_downloader.py`

**Interfaces:**
- Produces: `Downloader` class:
  - `clean_reel_url(raw_text: str) -> str | None`
  - `download_reel(reel_url: str, output_dir: Path) -> Path` (Returns absolute path to downloaded `.mp4`)

- [ ] **Step 1: Write failing tests for URL extraction and download**

```python
# tests/test_downloader.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.downloader import Downloader

def test_clean_reel_url():
    downloader = Downloader()
    
    # Standard reel
    url1 = "Check this out: https://www.instagram.com/reel/C-12345Abc/?igsh=MWx=="
    assert downloader.clean_reel_url(url1) == "https://www.instagram.com/reel/C-12345Abc/"
    
    # Reels URL variant
    url2 = "https://instagram.com/reels/C-12345Abc"
    assert downloader.clean_reel_url(url2) == "https://www.instagram.com/reel/C-12345Abc/"

    # Regular post variant
    url3 = "https://www.instagram.com/p/C-12345Abc/"
    assert downloader.clean_reel_url(url3) == "https://www.instagram.com/p/C-12345Abc/"

    # Non-instagram URL
    assert downloader.clean_reel_url("https://youtube.com/watch?v=123") is None

@patch("modules.downloader.YoutubeDL")
def test_download_reel_success(mock_ydl_class, tmp_path):
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.return_value = {"id": "C-12345Abc", "ext": "mp4"}
    
    expected_file = tmp_path / "C-12345Abc.mp4"
    expected_file.write_text("fake video content")
    
    downloader = Downloader()
    result_path = downloader.download_reel("https://www.instagram.com/reel/C-12345Abc/", output_dir=tmp_path)
    
    assert result_path.exists()
    assert result_path.name == "C-12345Abc.mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_downloader.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'modules.downloader')

- [ ] **Step 3: Implement modules/downloader.py**

```python
# modules/downloader.py
import re
from pathlib import Path
from yt_dlp import YoutubeDL

REEL_REGEX = re.compile(r'https?://(?:www\.)?instagram\.com/(?:reel|reels|p)/([a-zA-Z0-9_-]+)')

class Downloader:
    def __init__(self):
        pass

    def clean_reel_url(self, raw_text: str) -> str | None:
        match = REEL_REGEX.search(raw_text)
        if not match:
            return None
        shortcode = match.group(1)
        # Determine if it was a /p/ or /reel/
        if "/p/" in match.group(0):
            return f"https://www.instagram.com/p/{shortcode}/"
        return f"https://www.instagram.com/reel/{shortcode}/"

    def download_reel(self, reel_url: str, output_dir: Path) -> Path:
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
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(reel_url, download=True)
            video_id = info.get("id")
            # Look for the generated mp4 file
            expected_mp4 = output_dir / f"{video_id}.mp4"
            if expected_mp4.exists():
                return expected_mp4
            
            # Search for any file matching video_id in output_dir
            for f in output_dir.glob(f"{video_id}.*"):
                return f
                
            raise FileNotFoundError(f"Failed to locate downloaded file for reel ID {video_id}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_downloader.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/downloader.py tests/test_downloader.py
git commit -m "feat: add yt-dlp downloader module and URL cleaner"
```

---

### Task 4: Multimodal AI Analyzer Module (`Gemini 2.0 Flash`)

**Files:**
- Create: `modules/analyzer.py`
- Create: `tests/test_analyzer.py`

**Interfaces:**
- Produces:
  - `ReelAnalysisResult` (Pydantic model)
  - `ReelAnalyzer` class:
    - `analyze_video(video_path: Path) -> ReelAnalysisResult`

- [ ] **Step 1: Write failing tests for analyzer schema and video upload flow**

```python
# tests/test_analyzer.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.analyzer import ReelAnalyzer, ReelAnalysisResult

def test_pydantic_schema_validation():
    data = {
        "title": "Protein Pancakes",
        "category": "Recipes & Food",
        "tags": ["#recipes", "#breakfast"],
        "tldr": "Quick 3-ingredient protein pancakes recipe.",
        "key_takeaways": ["Use banana for moisture", "Cook on medium-low heat"],
        "detailed_steps": ["Mash banana", "Mix eggs and protein powder", "Pan fry for 2 min each side"],
        "key_timestamps": [5.2, 14.0],
        "timestamp_labels": ["Ingredients mix", "Final fluffy stack"]
    }
    result = ReelAnalysisResult.model_validate(data)
    assert result.title == "Protein Pancakes"
    assert result.category == "Recipes & Food"
    assert len(result.key_timestamps) == 2

@patch("modules.analyzer.genai.Client")
def test_analyze_video_mock(mock_client_class, tmp_path):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    # Mock file upload
    mock_file = MagicMock()
    mock_file.name = "files/test12345"
    mock_client.files.upload.return_value = mock_file
    
    # Mock model response
    mock_response = MagicMock()
    mock_response.text = """
    {
        "title": "Clean Architecture in Python",
        "category": "Tech & Coding",
        "tags": ["#python", "#architecture"],
        "tldr": "How to structure modular Python apps.",
        "key_takeaways": ["Separate domains from persistence", "Use dependency inversion"],
        "detailed_steps": ["Create core entities", "Define ports/interfaces", "Implement adapters"],
        "key_timestamps": [10.0, 25.5],
        "timestamp_labels": ["Layer diagram", "Code example"]
    }
    """
    mock_client.models.generate_content.return_value = mock_response
    
    video_file = tmp_path / "test.mp4"
    video_file.write_text("dummy")
    
    analyzer = ReelAnalyzer(api_key="fake_key")
    result = analyzer.analyze_video(video_file)
    
    assert result.title == "Clean Architecture in Python"
    assert result.category == "Tech & Coding"
    mock_client.files.delete.assert_called_once_with(name="files/test12345")
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

CategoryType = Literal[
    "Tech & Coding",
    "Recipes & Food",
    "Fitness & Health",
    "Finance & Investing",
    "Books & Learning",
    "Creative & Design",
    "General & Other"
]

class ReelAnalysisResult(BaseModel):
    title: str = Field(description="Descriptive, engaging title of the reel content")
    category: CategoryType = Field(description="Most accurate matching topic category")
    tags: list[str] = Field(description="List of 3-5 searchable hashtags with # prefix")
    tldr: str = Field(description="One or two-sentence executive summary")
    key_takeaways: list[str] = Field(description="Bullet points of the main insights or tips")
    detailed_steps: list[str] = Field(default_factory=list, description="Step-by-step instructions, ingredients, or methods if applicable")
    key_timestamps: list[float] = Field(description="2 to 4 exact timestamps in seconds where key visual details, diagrams, or final results appear")
    timestamp_labels: list[str] = Field(default_factory=list, description="Brief description of what is shown at each timestamp")

ANALYSIS_SYSTEM_INSTRUCTION = """
You are an expert multimedia analyst and knowledge extractor.
Your task is to analyze the provided Instagram Reel (both video visuals and spoken audio/transcription) and extract clear, actionable, organized knowledge.
1. Identify the core theme, category, and create a concise title.
2. Formulate a 1-2 sentence TL;DR.
3. List the top key actionable takeaways.
4. If it's a recipe, workout, or tutorial, provide step-by-step instructions.
5. Identify 2 to 4 exact moments (timestamps in seconds, e.g. 8.5) where the most valuable visual information appears (e.g. ingredients list, architecture diagram, code screenshot, final completed result).
Always return strict JSON conforming to the schema.
"""

class ReelAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def analyze_video(self, video_path: Path) -> ReelAnalysisResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Upload file using Google GenAI Files API
        uploaded_file = self.client.files.upload(file=video_path)
        
        # Wait for file processing if needed
        while uploaded_file.state == "PROCESSING":
            time.sleep(2)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    uploaded_file,
                    "Analyze this reel video and audio thoroughly and return the structured JSON knowledge."
                ],
                config=types.GenerateContentConfig(
                    system_instruction=ANALYSIS_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=ReelAnalysisResult,
                    temperature=0.2,
                )
            )
            
            result_data = json.loads(response.text)
            return ReelAnalysisResult.model_validate(result_data)
        finally:
            # Clean up uploaded file on Google AI servers
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add modules/analyzer.py tests/test_analyzer.py
git commit -m "feat: add Gemini 2.0 Flash multimodal analyzer module"
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

- [ ] **Step 1: Write failing tests for frame extraction and cleanup**

```python
# tests/test_frame_extractor.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.frame_extractor import FrameExtractor

@patch("subprocess.run")
def test_extract_frames_calls_ffmpeg(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    
    video_file = tmp_path / "test.mp4"
    video_file.write_text("fake video")
    
    extractor = FrameExtractor()
    timestamps = [5.5, 12.0]
    
    output_frames = extractor.extract_frames(video_file, timestamps, output_dir=tmp_path / "frames")
    
    assert len(output_frames) == 2
    assert mock_run.call_count == 2
    # Verify command contains timestamp
    cmd_1 = mock_run.call_args_list[0][0][0]
    assert "-ss" in cmd_1
    assert "5.5" in cmd_1

def test_cleanup_files(tmp_path):
    f1 = tmp_path / "img1.jpg"
    f2 = tmp_path / "img2.jpg"
    f1.write_text("data")
    f2.write_text("data")
    
    extractor = FrameExtractor()
    extractor.cleanup_files([f1, f2])
    
    assert not f1.exists()
    assert not f2.exists()
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
    def __init__(self):
        pass

    def extract_frames(self, video_path: Path, timestamps: list[float], output_dir: Path) -> list[Path]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_paths: list[Path] = []
        video_stem = video_path.stem

        # Limit to max 4 highlight frames to prevent album spam
        for idx, ts in enumerate(timestamps[:4]):
            output_frame = output_dir / f"{video_stem}_frame_{idx}_{int(ts)}s.jpg"
            # ffmpeg command: seek to timestamp, grab 1 frame with high quality
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
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                extracted_paths.append(output_frame)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If ffmpeg fails on a particular timestamp, continue to next
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
git commit -m "feat: add ffmpeg keyframe snapshot extractor module"
```

---

### Task 6: Telegram Publisher & Topic Router

**Files:**
- Create: `modules/publisher.py`
- Create: `tests/test_publisher.py`

**Interfaces:**
- Produces: `TelegramPublisher` class:
  - `format_summary_html(analysis: ReelAnalysisResult, original_url: str) -> str`
  - `publish_reel(analysis: ReelAnalysisResult, frame_paths: list[Path], original_url: str) -> int | None` (returns thread_id where posted)

- [ ] **Step 1: Write failing tests for HTML summary formatter and publisher router**

```python
# tests/test_publisher.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from modules.analyzer import ReelAnalysisResult
from modules.publisher import TelegramPublisher

@pytest.fixture
def sample_analysis():
    return ReelAnalysisResult(
        title="10-Minute Overnight Oats",
        category="Recipes & Food",
        tags=["#Recipes", "#Nutrition"],
        tldr="High protein meal prep breakfast recipe.",
        key_takeaways=["No cooking required", "35g of protein per serving"],
        detailed_steps=["Mix 50g oats with 150ml milk", "Stir in protein powder", "Refrigerate overnight"],
        key_timestamps=[5.0],
        timestamp_labels=["Finished jar"]
    )

def test_format_summary_html_contains_reel_link_at_bottom(sample_analysis):
    publisher = TelegramPublisher(bot_token="fake_token", group_chat_id="-100123", topic_map={})
    url = "https://www.instagram.com/reel/C-test123/"
    
    html = publisher.format_summary_html(sample_analysis, original_url=url)
    
    assert "🎬 <b>10-Minute Overnight Oats</b>" in html
    assert "📌 <b>TL;DR:</b>" in html
    assert "• No cooking required" in html
    assert "• 1. Mix 50g oats with 150ml milk" in html
    assert "#Recipes #Nutrition" in html
    # Ensure reel link is at the bottom
    assert f'🔗 <b>Original Reel:</b> <a href="{url}">{url}</a>' in html

@pytest.mark.asyncio
async def test_publish_reel_sends_media_and_message(sample_analysis, tmp_path):
    mock_bot = AsyncMock()
    publisher = TelegramPublisher(bot_token="fake_token", group_chat_id="-100123", topic_map={"Recipes & Food": 42})
    publisher.bot = mock_bot
    
    f1 = tmp_path / "f1.jpg"
    f1.write_text("fake image")
    
    url = "https://www.instagram.com/reel/C-test123/"
    thread_id = await publisher.publish_reel(sample_analysis, frame_paths=[f1], original_url=url)
    
    assert thread_id == 42
    mock_bot.send_media_group.assert_awaited_once()
    mock_bot.send_message.assert_awaited_once()
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
from modules.analyzer import ReelAnalysisResult

class TelegramPublisher:
    def __init__(self, bot_token: str, group_chat_id: str | int, topic_map: dict[str, int | None] | None = None):
        self.bot = Bot(token=bot_token)
        self.group_chat_id = group_chat_id
        self.topic_map = topic_map or {}

    def get_thread_id(self, category: str) -> int | None:
        return self.topic_map.get(category)

    def format_summary_html(self, analysis: ReelAnalysisResult, original_url: str) -> str:
        safe_title = html.escape(analysis.title)
        safe_tldr = html.escape(analysis.tldr)
        
        takeaways_html = "\n".join([f"• {html.escape(t)}" for t in analysis.key_takeaways])
        
        steps_block = ""
        if analysis.detailed_steps:
            steps_lines = "\n".join([f"• {i+1}. {html.escape(s)}" for i, s in enumerate(analysis.detailed_steps)])
            steps_block = f"\n\n📝 <b>Steps / Ingredients:</b>\n{steps_lines}"
            
        tags_line = " ".join(analysis.tags)
        safe_url = html.escape(original_url)

        return (
            f"🎬 <b>{safe_title}</b>\n\n"
            f"📌 <b>TL;DR:</b>\n{safe_tldr}\n\n"
            f"⚡ <b>Key Highlights:</b>\n{takeaways_html}"
            f"{steps_block}\n\n"
            f"🏷️ <i>{tags_line}</i>\n\n"
            f'🔗 <b>Original Reel:</b> <a href="{safe_url}">{safe_url}</a>'
        )

    async def publish_reel(self, analysis: ReelAnalysisResult, frame_paths: list[Path], original_url: str) -> int | None:
        thread_id = self.get_thread_id(analysis.category)
        
        # 1. Send photos if any exist
        existing_frames = [p for p in frame_paths if Path(p).exists()]
        if existing_frames:
            media = []
            files_to_close = []
            try:
                for idx, frame in enumerate(existing_frames[:4]):
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

        # 2. Send formatted summary HTML note
        text_content = self.format_summary_html(analysis, original_url)
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
git commit -m "feat: add Telegram publisher module with topic routing and HTML formatting"
```

---

### Task 7: End-to-End Orchestrator Pipeline, Telegram Listener & CLI Runner

**Files:**
- Create: `pipeline.py`
- Create: `main.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `ReelPipeline` class:
  - `process_reel(reel_url: str) -> tuple[bool, str]`
- Entrypoints:
  - `main.py` (Telegram bot polling daemon)
  - `main.py --url <reel_url>` (Immediate CLI test processing)

- [ ] **Step 1: Write failing test for ReelPipeline orchestration**

```python
# tests/test_pipeline.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from pipeline import ReelPipeline
from modules.analyzer import ReelAnalysisResult

@pytest.mark.asyncio
async def test_pipeline_execution_flow(tmp_path):
    # Mock settings & dependencies
    settings = MagicMock()
    settings.TEMP_DIR = tmp_path
    
    mock_db = MagicMock()
    mock_db.is_processed.return_value = False
    
    mock_downloader = MagicMock()
    video_file = tmp_path / "sample.mp4"
    video_file.write_text("fake video")
    mock_downloader.clean_reel_url.return_value = "https://www.instagram.com/reel/C-123/"
    mock_downloader.download_reel.return_value = video_file
    
    mock_analyzer = MagicMock()
    analysis = ReelAnalysisResult(
        title="Test Reel",
        category="Tech & Coding",
        tags=["#test"],
        tldr="Test tldr",
        key_takeaways=["Test takeaway"],
        key_timestamps=[1.0]
    )
    mock_analyzer.analyze_video.return_value = analysis
    
    mock_extractor = MagicMock()
    frame_file = tmp_path / "frame.jpg"
    frame_file.write_text("fake frame")
    mock_extractor.extract_frames.return_value = [frame_file]
    
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
    
    success, msg = await pipeline.process_reel("https://www.instagram.com/reel/C-123/")
    
    assert success is True
    assert "Test Reel" in msg
    mock_downloader.download_reel.assert_called_once()
    mock_analyzer.analyze_video.assert_called_once_with(video_file)
    mock_extractor.extract_frames.assert_called_once()
    mock_publisher.publish_reel.assert_awaited_once()
    mock_db.add_reel.assert_called_once()
    mock_extractor.cleanup_files.assert_called_once()
    # Video file should have been cleaned up
    assert not video_file.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'pipeline')

- [ ] **Step 3: Implement pipeline.py and main.py**

Create `pipeline.py`:
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
        self.db = db or ReelDatabase(settings.DATA_DIR / "reels.db")
        self.downloader = downloader or Downloader()
        self.analyzer = analyzer or ReelAnalyzer(api_key=settings.GEMINI_API_KEY)
        self.extractor = extractor or FrameExtractor()
        self.publisher = publisher or TelegramPublisher(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            group_chat_id=settings.TELEGRAM_GROUP_CHAT_ID,
            topic_map=settings.topic_map
        )
        self._lock = asyncio.Lock()

    async def process_reel(self, raw_input: str) -> tuple[bool, str]:
        async with self._lock:
            clean_url = self.downloader.clean_reel_url(raw_input)
            if not clean_url:
                return False, "Not a valid Instagram Reel or Post URL."

            if self.db.is_processed(clean_url):
                return True, f"Reel already processed previously: {clean_url}"

            video_dir = self.settings.TEMP_DIR / "videos"
            frames_dir = self.settings.TEMP_DIR / "frames"
            video_file: Path | None = None
            frame_paths: list[Path] = []

            try:
                # 1. Download
                video_file = self.downloader.download_reel(clean_url, output_dir=video_dir)

                # 2. Analyze via Gemini 2.0 Flash
                analysis = self.analyzer.analyze_video(video_file)

                # 3. Extract highlight photos
                frame_paths = self.extractor.extract_frames(
                    video_path=video_file,
                    timestamps=analysis.key_timestamps,
                    output_dir=frames_dir
                )

                # 4. Publish to Telegram
                thread_id = await self.publisher.publish_reel(
                    analysis=analysis,
                    frame_paths=frame_paths,
                    original_url=clean_url
                )

                # 5. Record in database
                self.db.add_reel(
                    reel_url=clean_url,
                    title=analysis.title,
                    category=analysis.category,
                    thread_id=thread_id
                )

                return True, f"Successfully processed: {analysis.title} -> Topic: {analysis.category}"

            except Exception as e:
                return False, f"Failed to process reel {clean_url}: {str(e)}"

            finally:
                # 6. Automatic cleanup of video and image files
                if video_file and video_file.exists():
                    try:
                        video_file.unlink()
                    except Exception:
                        pass
                if frame_paths:
                    self.extractor.cleanup_files(frame_paths)
```

Create `main.py`:
```python
# main.py
import argparse
import asyncio
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from config import Settings
from pipeline import ReelPipeline

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
        
    text = update.message.text
    if "instagram.com" not in text:
        return

    status_msg = await update.message.reply_text("⏳ Processing reel... Analyzing video & audio with Gemini...")
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    
    success, result = await pipeline.process_reel(text)
    
    # Update or remove status message
    if success:
        await status_msg.edit_text(f"✅ {result}")
    else:
        await status_msg.edit_text(f"⚠️ {result}")

def run_bot(pipeline: ReelPipeline, settings: Settings):
    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.bot_data["pipeline"] = pipeline
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    logger.info("Telegram Reel AI Bot is running. Send or share an Instagram reel link!")
    app.run_polling()

def main():
    parser = argparse.ArgumentParser(description="Instagram Reel AI Knowledge Agent")
    parser.add_argument("--url", type=str, help="Process a single Instagram Reel URL immediately via CLI")
    args = parser.parse_args()

    settings = Settings()
    pipeline = ReelPipeline(settings)

    if args.url:
        logger.info(f"Processing single URL: {args.url}")
        success, result = asyncio.run(pipeline.process_reel(args.url))
        print(f"[{'SUCCESS' if success else 'FAILED'}] {result}")
    else:
        run_bot(pipeline, settings)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`  
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`  
Expected: All tests PASS

- [ ] **Step 6: Commit changes**

```bash
git add pipeline.py main.py tests/test_pipeline.py
git commit -m "feat: implement end-to-end pipeline and main Telegram bot runner"
```

---

### Task 8: Verification & User Setup Guide

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create comprehensive README.md with 3-minute setup instructions**
- [ ] **Step 2: Verify complete test suite with coverage**
- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: add setup guide and instructions in README"
```
