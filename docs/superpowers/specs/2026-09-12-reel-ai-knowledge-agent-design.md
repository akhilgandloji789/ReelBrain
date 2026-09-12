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

### 3.2 Downloader Adapter (`modules/downloader.py`)
* **Tool:** `yt-dlp` invoked via isolated Python wrapper.
* **Failure Isolation:** Instagram frequently changes delivery endpoints. If `yt-dlp` fails:
  * Emits descriptive error with diagnostic hints (e.g. `yt-dlp update required` or `private reel`).
  * Gracefully informs user in Telegram without crashing the background service.
* **Storage & Cleanup:** Downloads to `temp/videos/{shortcode}.mp4`. File is guaranteed deleted immediately after Gemini upload and frame extraction.

### 3.3 Multimodal Analysis & Extensible Schema (`modules/analyzer.py`)
* **Model:** `gemini-2.0-flash` via official `google-genai` SDK.
* **Canonical Knowledge Schema:**
```json
{
  "title": "String (engaging, descriptive title)",
  "content_type": "recipe | workout | tech_coding | finance | book_ideas | travel | general",
  "tldr": "String (1-2 sentence executive summary)",
  "timeline": [
    { "timestamp": 0.0, "label": "Hook / Problem" },
    { "timestamp": 8.5, "label": "Ingredients / Tools" },
    { "timestamp": 22.0, "label": "Core Technique" },
    { "timestamp": 45.0, "label": "Final Result" }
  ],
  "claims_with_evidence": [
    {
      "claim": "Bake at 180°C for 20 minutes",
      "evidence_timestamp": 31.5,
      "confidence": "high"
    }
  ],
  "key_takeaways": ["Takeaway 1", "Takeaway 2"],
  "highlight_timestamps": [8.5, 45.0],
  "structured_data": {
    "recipe": {
      "prep_time_minutes": 10,
      "servings": 2,
      "ingredients": [{"item": "Rolled oats", "quantity": "50g"}],
      "instructions": ["Step 1...", "Step 2..."]
    },
    "tech": {
      "languages_tools": ["Python", "SQLite"],
      "code_snippets": ["pip install apscheduler"],
      "github_links": []
    },
    "workout": {
      "target_muscles": ["Quads", "Glutes"],
      "exercises": [{"name": "Bulgarian Split Squat", "sets": 3, "reps": "8-10"}]
    }
  },
  "tags": ["#Recipes", "#Nutrition", "#MealPrep"]
}
```

### 3.4 Keyframe Snapshot Extractor (`modules/frame_extractor.py`)
* **Tool:** `ffmpeg` via Python `subprocess`.
* Extracts high-resolution JPEG frames at the exact `highlight_timestamps` (max 3-4 images).
* Automatic deletion from `temp/frames/` immediately after Telegram media group upload.

### 3.5 Canonical Knowledge Database & Hybrid Search (`modules/storage.py` & `modules/search.py`)
* **Database:** SQLite `data/reelminds.db`
  * Table `reels`:
    - `id` (INTEGER PRIMARY KEY)
    - `source_id` (TEXT UNIQUE) — Instagram shortcode
    - `source_url` (TEXT)
    - `title` (TEXT)
    - `content_type` (TEXT)
    - `tldr` (TEXT)
    - `structured_data_json` (TEXT)
    - `claims_json` (TEXT)
    - `timeline_json` (TEXT)
    - `tags_csv` (TEXT)
    - `thread_id` (INTEGER)
    - `embedding_json` (BLOB/TEXT) — Gemini `text-embedding-004` vector (768 dims)
    - `created_at` (DATETIME)
    - `updated_at` (DATETIME)
  * Virtual Table `reels_fts` (SQLite FTS5):
    - Full-text search over `title`, `tldr`, `tags_csv`, `structured_data_json`.
* **Hybrid Retrieval Engine:**
  - Queries FTS5 for exact keyword matches.
  - Queries vector cosine similarity for conceptual/semantic matches (`"What should I do before gym?"` -> finds `"10-minute mobility routine"`).
  - Merges and ranks results (Reciprocal Rank Fusion / linear combination).
  - Feeds top 3-5 results to Gemini to generate conversational answers with source citations.

### 3.6 Telegram Presentation & Interactive Actions (`modules/publisher.py` & `main.py`)
* **Delivery:**
  1. **Photo Album:** `send_media_group` with the extracted highlight frames.
  2. **Evidence-Annotated HTML Note:**
     ```html
     🎬 <b>10-Minute High-Protein Overnight Oats</b>

     📌 <b>TL;DR:</b>
     Quick meal-prep recipe providing 35g protein without cooking.

     ⏱️ <b>Timeline:</b>
     • 00:08 — Ingredients breakdown
     • 00:32 — Mixing & consistency
     • 00:45 — Finished plated jar

     ⚡ <b>Key Highlights & Evidence:</b>
     • Base: 50g oats, 1 scoop vanilla whey <i>(00:08)</i>
     • Texture trick: 1 tbsp chia seeds creates pudding-like consistency <i>(00:20)</i>
     • Storage: Stays fresh up to 4 days refrigerated <i>(00:50)</i>

     🏷️ #Recipes #Nutrition #MealPrep

     🔗 <b>Original Reel:</b> https://instagram.com/reel/...
     ```
* **Interactive Command Hooks:**
  * `/ask <question>` — Conversational search across your saved second brain.
  * `/grocery` — (Reply to a recipe post) Formats an instant copy-pasteable shopping list.
  * `/code` — (Reply to a tech post) Extracts clean syntax-highlighted code blocks.
  * `/edit <old> -> <new>` — Enables user correction of extracted details (e.g. correcting a misheard quantity).
* **Sunday Action Review:**
  * Powered by `APScheduler` at 09:00 AM every Sunday.
  * Formats an actionable review of the week:
    ```
    🧠 YOUR REELMIND — SUNDAY REVIEW
    🍳 3 recipes saved (tap for grocery list)
    🏋️ 2 workouts saved (tap for routine)
    💻 4 tech ideas saved
    ⭐ Most recurring topic: AI Agents & Automation
    ```

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
