# ReelMind: The Instagram AI Second Brain

## Problem Statement
> *How might we turn ephemeral, unsearchable Instagram Reels into an organized, queryable personal knowledge vault with tailored actionable outputs—without cluttering personal messaging apps or risking account bans?*

## Recommended Direction
**ReelMind** is a Telegram-first personal intelligence agent that bridges Instagram to a permanent, queryable knowledge hub:

1. **Instant Smart Conversion:**  
   When you share a reel from Instagram to the bot, it downloads the video, analyzes visual frames and spoken audio with **Gemini 2.0 Flash**, and captures key photos via `ffmpeg`. It dynamically shapes the output:
   * 🍳 **Recipes:** Structured ingredient checklist + cooking steps + highlight photos of the dish.
   * 💻 **Tech & Coding:** Formatted code blocks, commands, and links.
   * 🏋️ **Workouts:** Exercises, target muscle groups, sets, and reps.
   * 📚 **Books & Ideas:** Core quotes, frameworks, and action items.  
   The bot posts the photo album and note into the matching **Telegram Forum Topic**, places the **original reel link at the bottom**, and **deletes the video file** from your machine.

2. **Conversational Retrieval ("Chat With Your Reels"):**  
   Every processed reel is indexed in a lightweight local SQLite Full-Text Search (FTS) engine. You can text your bot anytime in Telegram:  
   `"What was that protein pancake recipe with oats?"` or `/ask python tools for scraping`  
   The bot answers immediately in natural language, citing the exact notes and linking back to the reel.

3. **Weekly Sunday Digest:**  
   Every Sunday at 9:00 AM, the bot sends a compact, curated **"Sunday Morning Rollup"** highlighting the top 5 insights you saved during the week.

---

## Key Assumptions to Validate
- [ ] **Assumption 1 (Dynamic Schemas):** Gemini reliably extracts specific blocks (ingredients, code, reps) without hallucinating missing data. *(Test: Unit tests against sample recipe and coding reels).*
- [ ] **Assumption 2 (Natural Language Search):** SQLite FTS + Gemini can accurately answer user queries even when keywords are approximate or vague. *(Test: Query evaluation against indexed reel database).*
- [ ] **Assumption 3 (Zero Disk Bloat):** Video and frame files in `temp/` are reliably cleaned up after Telegram dispatch, keeping disk usage under 100MB permanently. *(Test: Automated cleanup verification).*

---

## MVP Scope

### What's In (MVP):
* Direct Share ingestion via Telegram Bot.
* `yt-dlp` video download with automatic cleanup.
* Gemini 2.0 Flash multimodal video+audio analysis with specialized schemas (Recipe, Tech, Workout, General).
* `ffmpeg` highlight snapshot extraction (up to 4 high-res photos) and cleanup.
* Telegram delivery: photo album + structured HTML note with the original reel link at the bottom.
* Telegram Forum Topic routing (`#Recipes`, `#Tech`, `#Fitness`, `#Finance`, `#General`).
* SQLite database with FTS5 search index.
* Conversational search query handler (`/ask <question>` or plain text questions).
* Background Sunday morning digest scheduler (`APScheduler`).

### What's Out (and Why):
* ❌ **No Instagram account bot logins:** 0% risk of account checkpoints or bans.
* ❌ **No permanent local MP4 storage:** Prevents your computer storage from filling up.
* ❌ **No separate web/mobile app:** Telegram already provides cross-platform cloud sync, search, and push notifications for $0.
* ❌ **No WhatsApp spam:** Keeps your personal WhatsApp 100% clean.

---

## Open Questions
- None blocking. Ready for implementation.
