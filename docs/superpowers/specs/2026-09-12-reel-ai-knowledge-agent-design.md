# Instagram Reel AI Knowledge Agent — Design Specification

**Date:** 2026-09-12  
**Status:** Approved for Implementation  
**Target Platform:** Python 3.10+ (Windows 11 / Cross-Platform)  
**Cost:** $0.00 (100% Free Tier)

---

## 1. Executive Summary & Problem Statement

Users frequently encounter informative, entertaining, or actionable Instagram Reels (recipes, coding tutorials, fitness workouts, book recommendations, productivity frameworks). However:
1. **Unsearchable Instagram Saves:** Saved reels on Instagram become a cluttered, unsearchable chronological list. Finding specific information requires re-watching dozens of videos.
2. **Messy Messaging Chats:** Forwarding reels to personal WhatsApp chats clutters conversations and photo galleries with unwieldy text blocks and loose images.
3. **Risk of Account Bans:** Traditional Instagram scraping bots that log in to secondary accounts risk account challenges, checkpoints, or suspensions by Meta.

### The Solution:
A zero-ban-risk, 100% free AI agent that bridges Instagram to a private **Telegram Knowledge Hub with Forum Topics**:
* **Trigger:** Share any reel from Instagram directly to your Telegram bot with two taps (`Share` → `Telegram`).
* **Processing:** The bot downloads the video, runs multimodal video & audio analysis using **Google Gemini 2.0 Flash**, and extracts crisp highlight photos using `ffmpeg`.
* **Output:** Formats the insights into a clean summary with bullet points, action items, tags, and a photo album, automatically routing it into categorized **Forum Topics** (e.g., `#Recipes`, `#Tech`, `#Fitness`, `#Finance`, `#General`).

---

## 2. High-Level Architecture

```
User on Instagram App
         │
         │ (Tap "Share" → Telegram Bot / Group)
         ▼
┌────────────────────────────────────────────────────────┐
│ Telegram Ingestion & Queue (python-telegram-bot)       │
│  - Instantly captures Reel URL                         │
│  - Sends status reaction / acknowledgment ("⏳ Working") │
│  - Enqueues job to sequential worker                   │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Video & Audio Downloader (yt-dlp)                      │
│  - Downloads MP4 file with audio to temp directory    │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Multimodal AI Analyzer (Gemini 2.0 Flash Free Tier)     │
│  - Uploads video to Google GenAI File API             │
│  - Analyzes visual frames + spoken audio simultaneously│
│  - Returns structured JSON: title, category, tags,     │
│    key takeaways, steps/recipes, highlight timestamps  │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Frame Extractor (ffmpeg)                               │
│  - Captures high-res JPEG photos at key timestamps    │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Telegram Topic Publisher                               │
│  - Maps category to group Forum Topic (thread_id)      │
│  - Sends photo album (sendMediaGroup)                  │
│  - Sends structured markdown summary with tags & links │
│  - Cleans up temporary video and image files           │
└────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Telegram Ingestion & Queue Manager
* **Library:** `python-telegram-bot` (async/await)
* **Ingestion:**
  * Handles direct messages to the bot or messages shared inside a private group.
  * Regex parses any Instagram URL variant: `https://(www\.)?instagram\.com/(reel|reels|p)/[a-zA-Z0-9_-]+`.
  * Sends an immediate feedback message: *"⏳ Processing reel... Analyzing video & audio"*.
* **Queue Safety:**
  * Processes one reel at a time per user to stay well within Google's 15 requests/minute rate limit.
  * If multiple reels are shared at once, they queue sequentially without dropping.

### 3.2 Downloader Module (`modules/downloader.py`)
* **Tool:** `yt-dlp` invoked via Python API.
* **Format:** Downloads optimized MP4 (`bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best`).
* **Storage:** Ephemeral directory `temp/videos/{reel_id}.mp4`.
* **Error Handling:** Gracefully handles private reels or unavailable posts, reporting a clear error to the user.

### 3.3 Multimodal AI Analysis (`modules/analyzer.py`)
* **Model:** `gemini-2.0-flash` via official `google-genai` SDK.
* **Input:** Native video file uploaded via Files API (`client.files.upload`).
* **Prompt Specification:**
  * System prompt instructs Gemini to act as an expert research summarizer.
  * Enforces strict Structured JSON Schema output:
```json
{
  "type": "object",
  "properties": {
    "title": { "type": "string" },
    "category": {
      "type": "string",
      "enum": [
        "Tech & Coding",
        "Recipes & Food",
        "Fitness & Health",
        "Finance & Investing",
        "Books & Learning",
        "Creative & Design",
        "General & Other"
      ]
    },
    "tags": {
      "type": "array",
      "items": { "type": "string" }
    },
    "tldr": { "type": "string" },
    "key_takeaways": {
      "type": "array",
      "items": { "type": "string" }
    },
    "detailed_steps": {
      "type": "array",
      "items": { "type": "string" }
    },
    "key_timestamps": {
      "type": "array",
      "items": { "type": "number" },
      "description": "List of 2 to 4 key seconds where vital visual details (results, diagrams, text) appear."
    },
    "timestamp_labels": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["title", "category", "tags", "tldr", "key_takeaways", "key_timestamps"]
}
```

### 3.4 Keyframe Snapshot Extractor (`modules/frame_extractor.py`)
* **Tool:** `ffmpeg` via Python `subprocess`.
* **Execution:**
  For each timestamp `ts` in `key_timestamps` (max 4 images):
  `ffmpeg -y -ss {ts} -i {video_path} -frames:v 1 -q:v 2 temp/frames/{reel_id}_{index}.jpg`
* Produces crisp JPEG images capturing visual tables, final recipes, code snippets, or diagrams shown in the reel.

### 3.5 Telegram Publisher & Topic Router (`modules/publisher.py`)
* **Topic Routing:**
  * Maps `category` to the corresponding Telegram group `message_thread_id`.
  * Pre-configured topic IDs can be defined in `.env` or auto-created if using a forum supergroup.
  * Defaults to General if thread ID is not mapped.
* **Message Delivery:**
  1. **Album:** Calls `bot.send_media_group` with the extracted highlight frames.
  2. **Summary Message:** Formatted in Telegram HTML:
     ```html
     🎬 <b>10-Minute High-Protein Overnight Oats</b>
     🔗 <a href="https://instagram.com/reel/...">Original Reel</a>

     📌 <b>TL;DR:</b>
     Quick meal-prep recipe providing 35g protein without cooking.

     ⚡ <b>Key Highlights:</b>
     • Base: 50g oats, 1 scoop vanilla whey, 150ml almond milk.
     • Texture trick: 1 tbsp chia seeds creates pudding-like consistency.
     • Storage: Stays fresh up to 4 days refrigerated.

     🏷️ #Recipes #Nutrition #MealPrep
     ```
  3. **Status Cleanup:** The initial *"⏳ Processing reel..."* message is deleted or updated to keep the chat clean.

### 3.6 Local Cache & Duplicate Prevention
* Lightweight SQLite database `data/reels.db`:
  * Table `reels`: `reel_url (TEXT PRIMARY KEY)`, `category (TEXT)`, `title (TEXT)`, `processed_at (DATETIME)`.
  * Prevents reprocessing the same reel if sent twice.

---

## 4. Directory Structure

```
c:/Akhil/Instagram/
├── .env.example
├── .env                       # TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, TELEGRAM_GROUP_ID
├── requirements.txt           # google-genai, python-telegram-bot, yt-dlp, pydantic
├── main.py                    # Application entrypoint & Telegram bot runner
├── config.py                  # Pydantic/dataclass settings validation
├── modules/
│   ├── __init__.py
│   ├── downloader.py          # yt-dlp video fetcher
│   ├── analyzer.py            # Gemini 2.0 Flash multimodal engine
│   ├── frame_extractor.py     # ffmpeg snapshot extractor
│   ├── publisher.py           # Telegram topic router & message sender
│   └── storage.py             # SQLite history & cache
├── temp/                      # Ephemeral video & frame storage (auto-cleaned)
└── data/                      # Persistent SQLite DB
```

---

## 5. Free-Tier Quota & Performance Analysis

| Metric | Google Gemini 2.0 Flash | Telegram Bot API |
| :--- | :--- | :--- |
| **Requests Limit** | 1,500 requests / day | Unlimited (30 msgs/sec) |
| **Token Limit** | 1,000,000 TPM | N/A |
| **Typical 1-min Reel Tokens** | ~18,000 tokens | N/A |
| **Max Free Reels / Day** | **1,500 reels** | Unlimited |
| **Financial Cost** | **$0.00 / month** | **$0.00 / month** |

---

## 6. Safety & Ban Risk Assessment

* **Instagram Ban Risk:** **0.0%**. No Instagram account is logged in programmatically. The user shares the link directly via standard Android/iOS share sheet to Telegram.
* **Gemini API Safety:** Backed by automatic rate-limiting queue and exponential backoff on `429 Too Many Requests`.
* **Data Privacy:** Data remains in the user's private Telegram group and Google AI Studio API session.

---

## 7. Verification & Testing Plan

1. **Unit Tests:**
   * URL extraction regex tests (handling reel URLs, query parameters, short links).
   * Schema validation tests for Gemini JSON output.
2. **Integration Verification:**
   * Test video download on a sample public reel with `yt-dlp`.
   * Test multimodal Gemini analysis with video upload.
   * Verify `ffmpeg` frame extraction at fractional timestamps.
   * Verify Telegram message & photo album delivery to specific topic thread.
3. **End-to-End Test:**
   * Send a test reel link into the Telegram bot chat.
   * Verify progress indicator, category classification, photo album, and topic delivery.
