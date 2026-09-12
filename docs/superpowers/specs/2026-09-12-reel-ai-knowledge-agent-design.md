# ReelMind: Personal AI Second Brain — Architectural Specification

**Date:** 2026-09-12 (Revised)  
**Status:** Approved for Implementation  
**Product Thesis:** A self-hosted, Telegram-native multimodal personal second brain that turns saved short-form content into structured, actionable knowledge.  
**Target Platform:** Python 3.10+ (Windows 11 / Linux / Docker)  
**Cost Model:** Designed to run within free tiers for personal self-hosted use (Gemini Free Tier, Telegram Bot API, SQLite).

---

## 1. Executive Summary & Strategic Positioning

### The Problem
Saved reels, videos, and posts on social platforms quickly become an unusable graveyard. When users want to retrieve a recipe, workout, programming tutorial, or framework saved days or weeks ago, search is non-existent, scrub-bar hunting in video is tedious, and information is lost.

### Why Generic "Reel Summarizers" Fail
1. **Market Saturation:** Standalone summarizers (Feedr, RecapIt, Instabrain, Reelnest) are commodities.
2. **Native Platform Threat:** Meta Muse and native platform AI are rolling out native summarization directly inside Instagram/WhatsApp.
3. **Lack of Grounded Trust:** Generic LLM summaries hallucinate ingredients, measurements, and code without source evidence.
4. **Passive vs. Actionable:** Summaries are passive reading; users actually need **actionable transformations** (recipes → grocery lists, workouts → exercise logs, tutorials → clean code snippets).

### The ReelMind Differentiation: Memory → Evidence → Retrieval → Action
ReelMind moves one abstraction layer above video summarization:
* **Telegram-Native Interface:** No new apps to download or configure. Ingestion happens via native mobile OS Share Sheet (`Share` → `Telegram`).
* **SQLite as Canonical Source of Truth:** Telegram is solely an interface/view; the underlying database maintains full structured knowledge objects, enabling future export, web dashboards, or cross-platform clients.
* **Evidence-Backed Extraction:** Every claim, ingredient, and instruction is indexed with exact video timeline timestamps (`Claim: Bake at 180°C. Evidence: 00:31-00:35`).
* **Hybrid Search (FTS5 + Semantic Embeddings):** Allows intuitive conceptual querying (`"What should I do before the gym?"` matches `"10-min pre-workout mobility routine"`).
* **Interactive Actions:** Direct command triggers in Telegram (`/ask`, `/grocery`, `/code`, `/edit`).
* **Zero Credential Dependency:** No Instagram bot logins or account cookies; eliminates account ban risk.

---

## 2. Decoupled System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   INGESTION LAYER                      │
│  Telegram Ingestion Adapter (Share Sheet Receiver)     │
│  - Parses & normalizes shortcode/URL                   │
│  - Deduplication Check (SQLite Hash & URL match)       │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│                 EXTRACTION ADAPTER                     │
│  Downloader Engine (yt-dlp) + Frame Grabbing (ffmpeg)   │
│  - Fetches MP4 video & audio stream to temp storage    │
│  - Captures high-res JPEG highlight frames at evidence  │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│            MULTIMODAL UNDERSTANDING ENGINE             │
│  Gemini 2.0 Flash + Extensible Content Schemas         │
│  - Classifies domain (Recipe, Workout, Tech, Ideas)    │
│  - Timeline extraction (Hook, Prep, Core, Climax)      │
│  - Grounded claims with exact evidence timestamps      │
│  - Typed structured_data payload                       │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│            CANONICAL KNOWLEDGE STORE (SOURCE OF TRUTH) │
│  SQLite Database + FTS5 Full-Text + Vector Embeddings   │
│  - Processed reels catalog                             │
│  - Normalized structured JSON entities                 │
│  - Text embeddings (text-embedding-004)                │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│               PRESENTATION & ACTION LAYER              │
│  Telegram Forum Topic Publisher & Interactive Bot      │
│  - Routes to topic thread (#Recipes, #Tech, etc.)      │
│  - Photo Album + Evidence-annotated HTML message       │
│  - Original Reel link at bottom                        │
│  - Interactive commands: /ask, /grocery, /code, /edit  │
│  - Sunday Action Review Digest (APScheduler)           │
│  - Automatic cleanup of temp MP4 & JPEG files          │
└────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Deduplication & Ingestion Manager (`modules/ingestion.py`)
* **URL Normalization:** Normalizes `instagram.com/reel/CODE`, `instagram.com/reels/CODE`, `instagram.com/p/CODE` to a canonical identifier.
* **Deduplication Check:** Before downloading, queries SQLite for the canonical ID.
  * If found: Replies in Telegram:
    ```
    ⚠️ Already saved on Sept 4, 2026 under 🍳 Recipes & Food!
    📌 "High-Protein Overnight Oats"
    🔗 View in topic thread
    ```
  * Prevents duplicate video downloads, redundant AI API calls, and channel clutter.

### 3.2 Downloader Adapter with Resilience (`modules/downloader.py`)
* **Tool:** `yt-dlp` wrapped with `tenacity` exponential backoff (3 attempts, 2s/4s/8s).
* **Failure Isolation:** Catches extraction errors cleanly, providing diagnostic hints without crashing the bot daemon.
* **Storage & Cleanup:** Downloads to `temp/videos/{shortcode}.mp4` and deletes immediately upon frame extraction and analysis.

### 3.3 Multimodal Analysis & Extensible Schema (`modules/analyzer.py`)
* **Model:** `gemini-2.0-flash` for multimodal video+audio understanding; `models/gemini-embedding-001` for vector embeddings (avoiding deprecated `text-embedding-004`).
* **Resilience:** Wrapped with `@retry` via `tenacity` with exponential backoff on `ResourceExhausted` (429) and network transport drops.
* **Canonical Knowledge Schema:**
  * Typed domain payload (`recipe`, `tech`, `workout`, `book_ideas`, `general`).
  * Video Timeline (`Hook`, `Prep`, `Technique`, `Result`).
  * Claims with exact evidence timestamps (`claim`, `evidence_timestamp`, `confidence`).

### 3.4 Keyframe Snapshot Extractor (`modules/frame_extractor.py`)
* **Tool:** `ffmpeg` via Python `subprocess`.
* Extracts high-resolution JPEG frames at the exact `highlight_timestamps` (max 3-4 images).
* Automatic deletion from `temp/frames/` immediately after Telegram media group upload.

### 3.5 Canonical Knowledge Store & Hybrid Search (`modules/storage.py` & `modules/search.py`)
* **Database:** SQLite `data/reelminds.db` with FTS5 virtual table.
* **Embeddings:** `gemini-embedding-001` vectors stored alongside records.
* **Synchronized Updates:** Whenever structured data or text is updated via `/edit`, the record's embedding is automatically regenerated and updated in the same transaction to maintain search index integrity.
* **Hybrid Retrieval:** Blends FTS5 keyword matching with cosine similarity on vector embeddings.

### 3.6 Telegram Presentation, Actions & Data Sovereignty (`modules/publisher.py`, `modules/actions.py`)
* **Delivery:** Photo album + Evidence-annotated HTML note with the original Reel link at the bottom.
* **Interactive Actions:**
  * `/ask <question>` — Conversational query answering with grounded citations.
  * `/grocery` — (Reply to recipe) Instant copy-pasteable shopping list.
  * `/code` — (Reply to tech post) Extracts clean syntax-highlighted code blocks.
  * `/edit <old> -> <new>` — Corrects AI inaccuracies and immediately triggers re-embedding.
  * `/export <md|json>` — Exports the entire knowledge base to a downloadable Markdown or JSON file sent directly in Telegram (Zero vendor lock-in).

### 3.7 Proactive Canary Health Check & Sunday Review (`modules/scheduler.py`)
* **Sunday Action Review:** Weekly Sunday 09:00 AM rollup of week's saved knowledge.
* **Canary Health Job:** Scheduled weekly test run (Tuesdays at 03:00 AM) that tests the pipeline against a stable test reel. If Instagram changes break `yt-dlp` or Gemini API changes, the bot proactively notifies the owner before a real user reel fails.

---

## 4. Operational, Legal & Cost Guardrails

1. **Credential & Ban Safety:**
   - Fact: **No Instagram credentials required.** ReelMind does not log into, scrape, or automate an Instagram user account. It only processes publicly accessible reel links shared directly by the user.
2. **Resource & Cost Expectations:**
   - **Cost:** Designed to run comfortably within free tiers for personal self-hosted use.
   - Google Gemini API free tier provides generous daily allowances, but is subject to Google terms and model lifecycle changes.
   - Temporary storage is strictly capped: videos and extracted frames are purged upon completion, keeping disk footprint under 100MB permanently.
3. **yt-dlp Resilience:**
   - Isolated inside an adapter with explicit exception catching and automated upgrade instructions in case Instagram changes public media delivery headers.

---

## 5. Phased Implementation Roadmap

* **Phase 1 (v1.0 Core MVP):**
  - Task 1: Configuration, Environment & Directory Scaffold
  - Task 2: Canonical SQLite Database with Schema, Migrations & Deduplication
  - Task 3: Downloader Adapter (`yt-dlp`) with Auto-Cleanup & Error Isolation
  - Task 4: Multimodal Analyzer with Typed Schemas, Timeline & Evidence Timestamps
  - Task 5: Keyframe Snapshot Extractor (`ffmpeg`)
  - Task 6: Telegram Publisher with Forum Topic Routing & Evidence-Annotated HTML Note
  - Task 7: End-to-End Pipeline & Telegram Bot Runner
* **Phase 2 (v1.1 Second Brain & Semantic Retrieval):**
  - Task 8: FTS5 + Embedding Hybrid Search Engine & `/ask` Conversational Query
* **Phase 3 (v1.2 Interactive Actions & Sunday Review):**
  - Task 9: Interactive Commands (`/grocery`, `/code`, `/edit`)
  - Task 10: Sunday Action Review Digest Scheduler (`APScheduler`)
* **Phase 4 (v1.3 Verification & Documentation):**
  - Task 11: Full Automated Test Suite & Self-Hosted Setup Guide
