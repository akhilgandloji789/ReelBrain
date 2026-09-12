# 🧠 ReelMind: Multimodal Personal AI Second Brain for Short-Form Video

> **Turn saved Instagram Reels into auditable, queryable, actionable personal knowledge.**  
> Built with Google Gemini 2.0 Flash, `models/gemini-embedding-001`, SQLite WAL + derived FTS5, and Telegram Bot API.

---

## 🌟 The Core Idea: Memory → Evidence → Retrieval → Action

Most people save dozens of Instagram Reels and TikToks every week — recipes, workout routines, coding tutorials, book recommendations, and travel spots — only for them to disappear into a digital graveyard.

Generic AI summarizers only produce surface-level text that nobody reads. **ReelMind** is fundamentally different:
1. **Memory:** Captures *why* you saved the video (`user_intent`), indexing personal context alongside video content.
2. **Evidence:** Grounded multimodal extraction. Every single claim, ingredient, or step is linked to an exact timestamp in the video (`⏱️ 00:31`), backed by keyframe photo snapshots. We don't ask you to trust arbitrary percentage numbers — you can scrub directly to the video timestamp to visually verify the evidence.
3. **Retrieval ("Ask My Memory"):** Hybrid search (SQLite FTS5 full-text + 768-dim vector embeddings) with conversational `/ask` that answers questions across all your saved reels citing exact video seconds.
4. **Action:** Turn knowledge into immediate action with streamlined **1-Tap Inline Buttons**:
   - `[🛒 Grocery List]` → Instant copyable grocery checklist from recipe reels.
   - `[💻 Copy Code]` → Instant syntax-highlighted code blocks from tech tutorials.
   - `[🔍 Ask Memory]` → Instant deep search over your personal second brain.
   - `[👍 Accurate] / [👎 Inaccurate]` → Clean 1-tap accuracy feedback.
5. **Data Sovereignty:** 100% self-hosted with SQLite; export your full brain anytime with `/export md` or `/export json`.
6. **Configurable Storage Policy:** Ephemeral video and frame files are purged immediately by default (`CLEANUP_TEMP=true`), keeping permanent disk footprint `< 100MB`. Toggle to `false` in `.env` for inspection during debugging.

---

## 🏗️ Architecture & Pipeline State Machine

```
[User shares Reel link (+ optional intent) in Telegram]
                         │
                         ▼
                [Idempotency Check]
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    [Already COMPLETED]       [New / Retrying Job]
    (Replies with topic info)           │
                                       ▼
                                 [RECEIVED]
                                       │
                                       ▼
                                [DOWNLOADING] (yt-dlp + Tenacity Retries + Circuit Breaker)
                                       │
                                       ▼
                                [DOWNLOADED]
                                       │
                                       ▼
                                 [ANALYZING] (Gemini 2.0 Flash Vision & Audio)
                                       │
                                       ▼
                                 [ANALYZED]
                                       │
                                       ▼
                             [EXTRACTING_FRAMES] (ffmpeg / OpenCV dual backend)
                                       │
                                       ▼
                                [PUBLISHING] (Telegram Topic, Photo Album, Inline Buttons)
                                       │
                                       ▼
                                 [INDEXING] (Atomic SQLite Transaction: Reels, Entities,
                                       │     Derived FTS5, gemini-embedding-001)
                                       ▼
                                [COMPLETED]
                                       │
                                       ▼
                                [AUTO-CLEANUP] (Purge temp video & keyframes)
```

---

## 🚀 Quickstart (3-Minute Setup)

### 1. Prerequisites
- Python 3.11+ (or Docker)
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))

### 2. Configuration
Clone the repository and copy the environment template:
```bash
git clone https://github.com/yourusername/reelmind.git
cd reelmind
cp .env.example .env
```

Edit `.env` with your credentials:
```env
TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
TELEGRAM_GROUP_CHAT_ID="-1001234567890"
GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
ALLOWED_TELEGRAM_USERS="your_telegram_user_id"
EMBEDDING_MODEL="models/gemini-embedding-001"
EMBEDDING_DIM=768
CANARY_REEL_URL="https://www.instagram.com/reel/example/"
```

### 3. Run via Docker Compose (Recommended)
```bash
docker-compose up -d
```

### 4. Run Locally
```bash
pip install -r requirements.txt
python main.py
```

### 5. CLI Mode (Test single reel without Telegram)
```bash
python main.py --url "https://www.instagram.com/reel/C-xyz123/ Try this for Sunday meal prep"
```

---

## 🕹️ Telegram Command & Interaction Reference

| Command / Action | Description |
|---|---|
| **Share Reel URL** | Paste or share an Instagram Reel URL with optional note (e.g. `https://instagram.com/reel/... Try this for meal prep`). Bot parses intent, extracts evidence, and publishes formatted note. |
| `/ask <query>` | Conversational retrieval over your saved reels ("Ask My Memory"). Cites exact video timestamps and links original source. |
| `/grocery <reel_id>` | Generates an interactive checklist of ingredients for recipes. |
| `/code <reel_id>` | Extracts clean, copyable code snippets from tech tutorials. |
| `/edit <id> <text>` | Corrects an extracted claim/ingredient and immediately triggers vector re-embedding. |
| `/status` | Real-time health dashboard: uptime, processing latency, indexed reels, entities, and actions. |
| `/canary` | On-demand proactive health check to verify yt-dlp, Gemini, and storage pipeline. |
| `/export [md\|json]` | Sends downloadable Markdown or JSON archive of your entire second brain. |
| **`[🛒 Grocery List]`** | 1-tap button on recipe notes for instant checklist generation. |
| **`[💻 Copy Code]`** | 1-tap button on tech notes for instant code retrieval. |
| **`[🔍 Ask Memory]`** | 1-tap button to start searching your saved reels. |
| **`[👍 Accurate]`** | Logs accuracy rating. |
| **`[👎 Inaccurate]`** | Logs inaccuracy and invites user correction via `/edit`. |

---

## ⏱️ Live Hackathon 3-Minute Demo Script

1. **0:00–0:20 (The Problem):**  
   *"We all save dozens of reels a week and forget them. Generic AI summarizes them, but summarization is a commodity. ReelMind is your personal second brain that captures not just what the video said, but why you saved it, and proves every claim with exact timestamps."*
2. **0:20–0:45 (Save with Intent):**  
   Share a Reel link live with a caption: `https://instagram.com/reel/... Try this for Sunday meal prep`  
   Bot immediately acknowledges: `⏳ Processing reel... Analyzing video & audio with Gemini...`
3. **0:45–1:15 (Visual Proof):**  
   Show the published note in Telegram: clean photo album, structured ingredients with clean evidence timestamps (`⏱️ 00:08`, `⏱️ 00:31`), and 1-tap action buttons.
4. **1:15–1:45 (Grounded `/ask`):**  
   Ask: `/ask what temperature do I bake this at?`  
   Bot responds: `Bake at 180°C (Evidence: ⏱️ 00:31)`. Scrub video to 00:31 on screen to demonstrate 100% auditable proof!
5. **1:45–2:15 (Zero-Friction Action):**  
   Tap the inline button `[🛒 Grocery List]` → checklist appears instantly without typing commands.
6. **2:15–2:40 (Personal Memory Query):**  
   Ask: `/ask what ideas did I save for Sunday meal prep?`  
   Bot answers matching your personal saved intent!
7. **2:40–3:00 (Reliability & Sovereignty):**  
   Run `/status` (real-time telemetry) → Run `/canary` (live health checks green) → Run `/export md` (data ownership download).

---

## 🧪 Running the Automated Test Suite

ReelMind has a comprehensive test suite covering all modules:
```bash
pytest -v
```

Output:
```
tests/test_actions.py ........ PASSED
tests/test_analyzer.py ....... PASSED
tests/test_config.py ......... PASSED
tests/test_downloader.py ..... PASSED
tests/test_frame_extractor.py  PASSED
tests/test_main.py ........... PASSED
tests/test_pipeline.py ....... PASSED
tests/test_publisher.py ...... PASSED
tests/test_scheduler.py ...... PASSED
tests/test_search.py ......... PASSED
tests/test_storage.py ........ PASSED

======================== 42 passed in 7.01s ========================
```

---

## 🛡️ Privacy & Reliability Guarantees
- **No Instagram Credentials:** ReelMind uses public unauthenticated link fetching. There is 0% risk of Instagram account flags or bans.
- **Whitelist Security:** Only Telegram user IDs configured in `ALLOWED_TELEGRAM_USERS` can trigger processing or queries.
- **Circuit Breaker:** If video fetching experiences consecutive transient failures, the circuit breaker opens for 2 minutes and alerts the administrator.
- **Truthful Evidence:** Model confidence is reported truthfully when returned; no fake default confidence numbers are injected.

---

## 📄 License
MIT License. Created with ❤️ for personal productivity and AI hackathons.
