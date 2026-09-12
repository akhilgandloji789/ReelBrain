# ReelMind: Personal AI Second Brain — Architectural Specification (Hackathon-Ready)

**Date:** 2026-09-12 (Enhanced Edition)  
**Status:** Approved for Implementation  
**Product Thesis:** A self-hosted, Telegram-native multimodal personal second brain that turns saved short-form content into structured, actionable knowledge, anchored on **Memory → Evidence → Retrieval → Action**.  
**Target Platform:** Python 3.11+ (Windows 11 / Linux / Docker)  
**Cost & Resource Model:** Designed to run comfortably within free tiers for personal self-hosted use (Gemini Free Tier, Telegram Bot API, SQLite).

---

## 1. Executive Summary & Strategic Positioning

### The Problem
Saved reels, shorts, and videos become an unsearchable digital graveyard. When a user wants to retrieve a recipe, workout routine, command, or framework saved days or weeks ago, search is non-existent, scrub-bar hunting in video is tedious, and information is lost.

### The Value Proposition: Prove It Isn't Hallucinating
Generic AI note-takers passively summarize text. ReelMind indexes video with **grounded provenance**:
* **Telegram-Native UX with Inline Keyboards:** Share sheet ingestion (`Share` → `Telegram`), with interactive buttons on every post (`[🛒 Grocery List]`, `[💻 Copy Code]`, `[🔍 Ask Question]`, `[✏️ Edit]`).
* **Canonical SQLite Source of Truth (WAL Mode):** Fully normalized schema (`reels`, `entities`, `embeddings`, `action_log`, `reels_fts`) with transactional integrity. Telegram is solely an interface/view.
* **Evidence Trust Model:** Every claim and ingredient carries a `(text, start_ts, end_ts, confidence)` tuple with linked frame snapshots. In `/ask` responses, every answer cites source timestamps with confidence scores.
* **Hybrid Search (FTS5 + `gemini-embedding-001`):** Blends BM25 keyword matching with vector cosine similarity and re-ranking for conceptual queries.
* **Zero Credential Dependency & Security Whitelist:** No Instagram bot credentials required (zero ban risk). Bot commands and ingestion are guarded by an allowed Telegram User/Chat ID whitelist.
* **Observability & Proactive Canaries:** In-memory telemetry (`/status`), circuit breakers for external services, and scheduled canary pipeline tests to catch `yt-dlp` breaks early.

---

## 2. System Architecture & Performance Budget

```
                    ┌─────────────────────────┐
   User shares      │ Telegram Bot + Webhook/ │
   Reel URL   ─────▶│ Polling + Auth Guard    │
                    └───────────┬─────────────┘
                                │ (Check Chat Whitelist & Rate Limits)
                    ┌───────────▼─────────────┐
                    │ Pipeline Orchestrator   │
                    │ + Circuit Breaker       │
                    └───────────┬─────────────┘
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                      ▼
┌──────────────────┐  ┌───────────────────┐  ┌──────────────────┐
│ Downloader       │  │ Multimodal Vision │  │ Keyframe Grabber │
│ (yt-dlp adapter) │  │ (Gemini 2.0 Flash)│  │ (ffmpeg)         │
│ + Tenacity Retry │  │ + Gemini Embed 001│  │ + Auto-Cleanup   │
└─────────┬────────┘  └─────────┬─────────┘  └─────────┬────────┘
          │                     │                      │
          └─────────────────────┼──────────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ Canonical SQLite DB     │  Atomic Transaction:
                    │ (WAL Mode, reels,       │  Reel + Entities +
                    │  entities, embeddings,  │  FTS5 + Action Log
                    │  reels_fts, action_log) │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                      ▼
┌──────────────────┐  ┌───────────────────┐  ┌──────────────────┐
│ Search Engine    │  │ Action Handler    │  │ Telegram Router  │
│ (Hybrid FTS5 +   │  │ (/grocery, /code, │  │ (Topic Threads,  │
│  Cosine Rerank)  │  │  /edit, /export)  │  │  Inline Buttons) │
└──────────────────┘  └───────────────────┘  └──────────────────┘
                                │
                      ┌─────────▼──────────┐
                      │ Telemetry & Canary │
                      │ (/status, APSched) │
                      └────────────────────┘
```

### Performance & Operational Budget
* **Max Reel Duration:** 90 seconds (longer media gracefully warned to protect rate limits).
* **Target Latency:** End-to-end processing < 25s for a 60s reel.
* **Frame Extraction Budget:** Maximum 4 high-res JPEG keyframes per reel (< 4MB Telegram payload).
* **Storage Footprint:** Ephemeral `temp/` directory strictly purged after processing (< 50MB permanent local footprint).
* **Rate Limits:** 15 RPM, 1,500 RPD budget enforced with sequential queue and circuit breakers.

---

## 3. Database Schema (SQLite — `modules/storage.py`)

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Canonical reel records
CREATE TABLE IF NOT EXISTS reels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    shortcode       TEXT UNIQUE NOT NULL,
    content_hash    TEXT,
    saved_at        TEXT NOT NULL,
    category        TEXT NOT NULL,              -- recipe | tech | workout | idea | travel | finance | other
    title           TEXT,
    raw_transcript  TEXT,
    status          TEXT NOT NULL DEFAULT 'pending', -- pending | processed | failed
    error_message   TEXT
);

-- Grounded claims and entities with evidence timestamps
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_id         INTEGER NOT NULL REFERENCES reels(id) ON DELETE CASCADE,
    entity_type     TEXT NOT NULL,              -- ingredient | instruction | code_snippet | tech_fact | exercise | idea_point
    text            TEXT NOT NULL,
    start_ts        REAL NOT NULL,              -- seconds into video
    end_ts          REAL,
    confidence      REAL DEFAULT 0.90,          -- model confidence score
    frame_path      TEXT,                       -- optional snapshot filename
    edited_by_user  INTEGER DEFAULT 0           -- 1 if modified via /edit
);

-- Vector embeddings (gemini-embedding-001, 768-dim)
CREATE TABLE IF NOT EXISTS embeddings (
    reel_id     INTEGER PRIMARY KEY REFERENCES reels(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,                  -- packed float32 array
    updated_at  TEXT NOT NULL
);

-- FTS5 Virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS reels_fts USING fts5(
    title, raw_transcript, entity_text,
    content='', tokenize='porter unicode61'
);

-- Audit and telemetry log
CREATE TABLE IF NOT EXISTS action_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_id     INTEGER REFERENCES reels(id),
    action_type TEXT NOT NULL,                  -- grocery | code | edit | ask | export
    payload     TEXT,                           -- JSON blob
    created_at  TEXT NOT NULL
);
```

---

## 4. Component Architecture & Improvements

### 4.1 Security & Access Guard (`modules/security.py`)
* Validates incoming Telegram `user_id` and `chat_id` against `ALLOWED_TELEGRAM_USERS` in `.env`.
* Rejects unauthorized users before any compute or external API calls are made.
* Safeguards `/export` and `/edit` commands from unauthorized triggers.

### 4.2 Downloader Adapter with Circuit Breaker (`modules/downloader.py`)
* `yt-dlp` wrapped with `tenacity` exponential backoff (3 attempts, 2s/4s/8s).
* Circuit breaker: after 3 consecutive failures, enters open state for 2 minutes and returns diagnostic advice.
* Parses canonical shortcode to reject duplicates before downloading.

### 4.3 Multimodal Understanding & Provenance (`modules/analyzer.py`)
* Model: Gemini 2.0/2.5 Flash for audio & visual reasoning.
* Strict JSON schema enforcing `entities` with `start_ts`, `end_ts`, and `confidence`.
* Embeddings: `models/gemini-embedding-001` (768-dim MRL output).

### 4.4 Hybrid Search & Re-ranking (`modules/search.py`)
* Keyword matching: SQLite FTS5 `MATCH` with BM25 ranking.
* Semantic matching: Cosine similarity across packed float32 embedding vectors.
* Re-ranking: Blends BM25 score (0.4) and semantic similarity (0.6), boosted by entity confidence.
* Grounded `/ask`: Responses cite timestamps and confidence scores (`[Claim] ⏱️ 00:31 | 95% confidence`), refusing ungrounded facts.

### 4.5 Telegram UI with Inline Keyboards (`modules/publisher.py`)
* Every processed reel published to its category Forum Topic includes:
  1. Photo album of evidence keyframes.
  2. Evidence-annotated HTML note with the original reel link at the bottom.
  3. Interactive Telegram **Inline Keyboard**:
     * `[🛒 Get Grocery List]` (for recipes)
     * `[💻 Copy Code]` (for tech tutorials)
     * `[🔍 Ask Question]`
     * `[✏️ Edit Correction]`

### 4.6 Telemetry & Proactive Health (`modules/metrics.py` & `modules/scheduler.py`)
* Real-time metrics: latency, success rate, total reels indexed, and storage size.
* `/status` command: displays real-time health dashboard in Telegram.
* Weekly Canary Health Job (Tuesdays at 03:00 AM): tests pipeline against a designated stable reel and alerts owner if `yt-dlp` breaks.
* On-demand `/canary` command for live hackathon demonstration of proactive health monitoring.
* Sunday Action Review Digest (Sundays at 09:00 AM).

### 4.7 Data Sovereignty & Offline Demo Fallback (`modules/actions.py`)
* `/export md` & `/export json`: instant downloadable data dumps sent as Telegram files.
* `--demo` CLI flag: executes against pre-packaged video/analysis fixtures in `tests/fixtures/` so the live stage demo cannot fail even with broken venue Wi-Fi.

---

## 5. Hackathon Winning Strategy: 3-Minute Stage Demo Script

1. **The 15-Second Hook:**  
   *"Every AI tool can summarize a video. None can prove the summary is true. ReelMind timestamps every claim back to the exact second in the video."*
2. **The Live Trigger (30s):**  
   Share a recipe or workout Reel live into Telegram → Bot immediately acknowledges → processes in < 20s → posts album + formatted note with timestamps into `#Recipes`.
3. **The Proof Moment (45s):**  
   Tap the inline button `[🔍 Ask Question]` or type `/ask what temperature do I bake this at?` → Bot replies: *"Bake at 180°C (Evidence: 00:31, Confidence: 96%)"*. Scrub the video to 00:31 on screen to visually prove zero hallucination!
4. **The Action Moment (30s):**  
   Tap `[🛒 Get Grocery List]` → instant copyable checklist appears.
5. **The Technical Defense (30s):**  
   Show SQLite canonical architecture, zero-credential ban-free safety, and one-click `/export` to Markdown.
