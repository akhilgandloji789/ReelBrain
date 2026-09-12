# ReelMind: Personal AI Second Brain — Architectural Specification (Master Edition)

**Date:** 2026-09-12 (Production & Hackathon Master Edition)  
**Status:** Approved for Implementation  
**Product Thesis:** A self-hosted, Telegram-native multimodal personal second brain that turns saved short-form content into structured, actionable knowledge, anchored on **Memory → Evidence → Retrieval → Action**.  
**Core Differentiation:** Bridges the gap between what a video says and *why you cared about it*, grounding every extracted claim in verifiable source timestamps.  
**Target Platform:** Python 3.11+ (Windows 11 / Linux / Docker)  
**Cost Model:** Designed to run comfortably within free tiers for personal self-hosted use (Gemini Free Tier, Telegram Bot API, SQLite).

---

## 1. Executive Summary & Strategic Positioning

### The Problem
Saved reels, shorts, and bookmarks become an unsearchable digital graveyard. When a user wants to retrieve a recipe, workout routine, command, or framework saved days or weeks ago, search is non-existent, scrub-bar hunting in video is tedious, and information is lost.

### Why "Summarization" is a Commodity Trap
Standalone AI video summarizers are rapidly commoditizing, while platforms (Meta Muse) are rolling out native summarization. ReelMind wins by operating one abstraction layer higher: **Personal Memory + Grounded Provenance + Actionable Transformation**.

### The ReelMind Architecture Pillars:
1. **Auditable Evidence (Confidence ≠ Truth):**
   * Claims, ingredients, and instructions carry `(text, start_ts, end_ts, confidence)`.
   * Confidence is nullable with no fake defaults. If the model didn't return a score, the UI displays *"Source timestamp verified"* rather than a fabricated percentage.
   * Responses in `/ask` cite source timestamps with evidence snippets.
2. **Personal Intent ("Why Did I Save This?"):**
   * Captures optional user notes/captions sent with the link (or via `/intent`).
   * Indexes user intent into SQLite, FTS5, and vector embeddings, enabling queries like `/ask what ideas did I save for improving ReelMind?`.
3. **Interactive Inline Keyboards & Feedback Loop:**
   * Inline buttons: `[🛒 Grocery List]` / `[💻 Copy Code]`, `[🔍 Ask]`, `[✏️ Edit]`, `[👍 Accurate]`, `[👎 Inaccurate]`.
   * User feedback and edits are logged to `action_log`, with edits atomically refreshing embeddings and FTS.
4. **Resilient State Machine & Idempotency:**
   * Explicit job state machine: `RECEIVED` → `DOWNLOADING` → `DOWNLOADED` → `ANALYZING` → `ANALYZED` → `EXTRACTING_FRAMES` → `PUBLISHING` → `INDEXING` → `COMPLETED`.
   * Idempotency checks based on shortcode hash prevent duplicate processing or duplicate Telegram notes.
5. **Transactional Storage & Startup Recovery:**
   * SQLite in WAL mode with derived FTS5 synchronization.
   * On startup, a cleanup routine purges orphaned temp files and recovers stalled jobs, guaranteeing `<100MB` permanent local disk footprint.
6. **Observability & Proactive Health:**
   * `/status`: Real-time telemetry dashboard (uptime, latency, indexed counts, error rates).
   * `/canary`: Live diagnostic check verifying Telegram, SQLite, Gemini API, ffmpeg, and yt-dlp.
7. **Data Sovereignty & Portability:**
   * `/export md` and `/export json` deliver full database dumps directly as Telegram attachments.

---

## 2. Database Schema (SQLite — `modules/storage.py`)

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Canonical reel records with state machine
CREATE TABLE IF NOT EXISTS reels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    shortcode       TEXT UNIQUE NOT NULL,       -- Primary idempotency key
    content_hash    TEXT,                        -- Secondary dedup hash
    saved_at        TEXT NOT NULL,               -- ISO timestamp
    category        TEXT NOT NULL,               -- recipe | tech | workout | idea | travel | finance | other
    title           TEXT,
    user_intent     TEXT,                        -- "Why I saved this" personal context
    raw_transcript  TEXT,
    status          TEXT NOT NULL DEFAULT 'RECEIVED', -- RECEIVED|DOWNLOADING|DOWNLOADED|ANALYZING|ANALYZED|EXTRACTING_FRAMES|PUBLISHING|INDEXING|COMPLETED|FAILED
    error_message   TEXT
);

-- Grounded claims and entities with evidence timestamps (NO fake default confidence)
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_id         INTEGER NOT NULL REFERENCES reels(id) ON DELETE CASCADE,
    entity_type     TEXT NOT NULL,               -- ingredient | instruction | code_snippet | tech_fact | exercise | idea_point
    text            TEXT NOT NULL,
    start_ts        REAL NOT NULL,               -- seconds into video
    end_ts          REAL,                        -- nullable end timestamp
    confidence      REAL,                        -- Model-reported confidence (nullable, no default)
    frame_path      TEXT,                        -- Keyframe snapshot filename
    edited_by_user  INTEGER DEFAULT 0            -- 1 if modified via /edit
);

-- Vector embeddings (Configurable model & dimension)
CREATE TABLE IF NOT EXISTS embeddings (
    reel_id     INTEGER PRIMARY KEY REFERENCES reels(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,                   -- e.g. 'models/gemini-embedding-001'
    dim         INTEGER NOT NULL,                -- e.g. 768
    vector      BLOB NOT NULL,                   -- packed float32 array
    updated_at  TEXT NOT NULL
);

-- FTS5 Virtual table (Derived index covering title, transcript, intent, and entities)
CREATE VIRTUAL TABLE IF NOT EXISTS reels_fts USING fts5(
    title,
    user_intent,
    raw_transcript,
    entity_text,
    content='',
    tokenize='porter unicode61'
);

-- Audit, feedback, and action log
CREATE TABLE IF NOT EXISTS action_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_id     INTEGER REFERENCES reels(id),
    action_type TEXT NOT NULL,                   -- grocery | code | edit | ask | export | feedback_thumb_up | feedback_thumb_down
    payload     TEXT,                            -- JSON blob of action details
    created_at  TEXT NOT NULL
);
```

---

## 3. Pipeline State Machine & Idempotency Flow

```
[User sends Reel URL (+ optional intent)]
                     │
                     ▼
             [Idempotency Check]
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[Already COMPLETED]       [New / Retrying Job]
(Reply: "Already saved")           │
                                   ▼
                             [RECEIVED]
                                   │
                                   ▼
                            [DOWNLOADING] (yt-dlp + Tenacity + Circuit Breaker)
                                   │
                                   ▼
                            [DOWNLOADED]
                                   │
                                   ▼
                             [ANALYZING] (Gemini 2.0 Flash Multimodal Vision & Audio)
                                   │
                                   ▼
                             [ANALYZED]
                                   │
                                   ▼
                         [EXTRACTING_FRAMES] (ffmpeg keyframes at evidence timestamps)
                                   │
                                   ▼
                            [PUBLISHING] (Telegram Topic, Photo Album, Inline Buttons)
                                   │
                                   ▼
                             [INDEXING] (Atomic SQLite Transaction: Reels, Entities,
                                   │     FTS5 Derived Sync, Gemini-Embedding-001)
                                   ▼
                            [COMPLETED]
                                   │
                                   ▼
                            [AUTO-CLEANUP] (Purge temp video & frames)
```

---

## 4. Grounded Evidence Trust Model

* **Confidence Reporting:**  
  When Gemini returns confidence: `⏱️ 00:31 | Confidence: 94%`.  
  When confidence is null: `⏱️ 00:31 | Evidence verified in video`.  
  *No hardcoded fake defaults are ever inserted.*
* **Conversational `/ask` Provenance:**  
  The retrieval engine passes retrieved evidence to Gemini with the instruction:  
  *"Answer using ONLY the provided evidence. Cite the exact start_ts and confidence score for each fact. Refuse to answer claims not supported by the evidence."*

---

## 5. Hackathon 3-Minute Live Demo Script

1. **0:00–0:20 (The Problem):**  
   *"We all save dozens of reels a week and forget them. Generic AI summarizes them, but summarization is a commodity. ReelMind is your personal second brain that captures not just what the video said, but why you saved it, and proves every claim with exact timestamps."*
2. **0:20–0:40 (Save with Intent):**  
   Share a Reel link live with a caption: `https://instagram.com/reel/... Try this for meal prep`  
   Bot immediately acknowledges: `⏳ Indexing into #Recipes...`
3. **0:40–1:10 (Visual Proof):**  
   Show the published note in Telegram: clean photo album, structured ingredients with evidence timestamps (`⏱️ 00:08`, `⏱️ 00:31`), and interactive buttons.
4. **1:10–1:40 (Grounded `/ask`):**  
   Ask: `/ask what temperature do I bake this at?`  
   Bot responds: `Bake at 180°C (Evidence: 00:31, Confidence: 94%)`. Scrub video to 00:31 on screen to demonstrate auditable proof!
5. **1:40–2:10 (Instant Action):**  
   Tap the inline button `[🛒 Get Grocery List]` → checklist appears instantly.
6. **2:10–2:35 (Personal Memory Query):**  
   Ask: `/ask what ideas did I save for meal prep?`  
   Bot answers matching your personal saved intent!
7. **2:35–3:00 (Engineering & Sovereignty):**  
   Run `/status` (real-time telemetry) → Run `/canary` (live health checks green) → Run `/export md` (data ownership download).
