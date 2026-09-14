import html
import logging
from typing import Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from modules.storage import ReelDatabase

logger = logging.getLogger("scheduler")


class SchedulerService:
    def __init__(
        self,
        db: ReelDatabase,
        bot: Any,
        chat_id: str | int,
        pipeline: Any = None,
        canary_url: str | None = None,
        channel_check_interval_mins: int = 30,
        instagram_receiver: Any = None
    ):
        self.db = db
        self.bot = bot
        self.chat_id = chat_id
        self.pipeline = pipeline
        self.canary_url = canary_url
        self.channel_check_interval_mins = channel_check_interval_mins
        self.instagram_receiver = instagram_receiver
        self.scheduler = AsyncIOScheduler()

    def build_weekly_digest(self) -> str | None:
        recent = []
        if hasattr(self.db, "get_all_week_reels_with_entities"):
            try:
                res = self.db.get_all_week_reels_with_entities(days=7)
                if isinstance(res, list) and res:
                    recent = res
            except Exception:
                pass

        if not recent and hasattr(self.db, "get_recent_reels"):
            try:
                res = self.db.get_recent_reels(days=7)
                if isinstance(res, list) and res:
                    recent = res
            except Exception:
                pass

        if not recent:
            return None

        category_map = {
            "recipe": "🍳 Recipes",
            "tech": "💻 Tech & AI",
            "workout": "💪 Fitness",
            "idea": "💡 Ideas & Wisdom",
            "travel": "✈️ Travel",
            "finance": "💰 Finance",
            "other": "📌 Other",
        }

        by_cat: dict[str, list[dict[str, Any]]] = {}
        for r in recent:
            c = (r.get("category") or "other").lower()
            by_cat.setdefault(c, []).append(r)

        total_reels = len(recent)
        lines = [
            "🌙 <b>REELBRAIN — SUNDAY MASTER DIGEST</b>",
            "🧠 <b>YOUR REELMIND — SUNDAY REVIEW</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📅 <b>Weekly Executive Review: {total_reels} Reels Indexed</b>\n"
            "Here is all the knowledge captured this week across all topics:\n"
        ]

        for cat, reels in by_cat.items():
            cat_label = category_map.get(cat, f"📌 #{cat.title()}")
            lines.append(f"{cat_label} (<b>{len(reels)}</b>):")
            for r in reels:
                title = html.escape(r.get("title") or "Untitled Reel")
                url = html.escape(r.get("url") or "#")
                lines.append(f"• <a href='{url}'><b>{title}</b></a>")
                
                tldr = r.get("raw_transcript") or ""
                if tldr:
                    tldr_clean = html.escape(tldr.replace("\n", " ").strip())
                    if len(tldr_clean) > 90:
                        tldr_clean = tldr_clean[:87] + "..."
                    lines.append(f"  ↳ <i>{tldr_clean}</i>")
                elif r.get("entities"):
                    top_ents = [html.escape(e["text"]) for e in r["entities"][:2]]
                    lines.append(f"  ↳ <i>{'; '.join(top_ents)}</i>")
            lines.append("")

        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("💡 <i>Ask your second brain anything with /ask, or explore topics with /filter.</i>")
        
        digest_text = "\n".join(lines)
        if len(digest_text) > 4000:
            compact_lines = [
                "🌙 <b>REELBRAIN — SUNDAY MASTER DIGEST</b>",
                "━━━━━━━━━━━━━━━━━━━━",
                f"📅 <b>Weekly Executive Review: {total_reels} Reels Indexed</b>\n"
            ]
            for cat, reels in by_cat.items():
                cat_label = category_map.get(cat, f"📌 #{cat.title()}")
                compact_lines.append(f"{cat_label} (<b>{len(reels)}</b>):")
                for r in reels:
                    title = html.escape(r.get("title") or "Untitled Reel")
                    url = html.escape(r.get("url") or "#")
                    compact_lines.append(f"• <a href='{url}'>{title}</a>")
                compact_lines.append("")
            compact_lines.append("━━━━━━━━━━━━━━━━━━━━")
            compact_lines.append("💡 <i>Use /ask or /filter to explore all detailed notes.</i>")
            digest_text = "\n".join(compact_lines)

        return digest_text

    async def send_weekly_digest(self) -> None:
        digest = self.build_weekly_digest()
        if digest:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=digest,
                    parse_mode="HTML",
                    disable_web_page_preview=True
                )
            except Exception as e:
                logger.error(f"Failed to send weekly digest: {e}")

    async def run_canary_test(self) -> tuple[bool, str]:
        if not self.pipeline or not self.canary_url:
            return True, "No canary URL configured."
        try:
            success, msg, _ = await self.pipeline.process_url(self.canary_url)
            if not success:
                alert_text = f"🚨 <b>Canary Alert:</b> Pipeline health check failed!\nDetails: {msg}"
                try:
                    await self.bot.send_message(chat_id=self.chat_id, text=alert_text, parse_mode="HTML")
                except Exception:
                    pass
                return False, msg
            return True, "Canary health check passed."
        except Exception as e:
            alert_text = f"🚨 <b>Canary Alert:</b> Exception during health check:\n{str(e)}"
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=alert_text, parse_mode="HTML")
            except Exception:
                pass
            return False, str(e)

    async def scan_tracked_channels(self) -> dict[str, int]:
        channels = self.db.get_tracked_channels(active_only=True)
        total_new = 0
        if not self.pipeline:
            return {"scanned_channels": len(channels), "new_reels_processed": 0}

        for ch in channels:
            handle = ch["handle"]
            try:
                downloader = getattr(self.pipeline, "downloader", None)
                if not downloader:
                    continue
                reel_urls = downloader.get_channel_reels(handle, limit=3)
            except Exception as e:
                logger.warning(f"Failed to fetch reels for creator @{handle}: {e}")
                continue

            latest_shortcode = None
            for reel_url in reel_urls:
                shortcode, canonical_url, _ = self.pipeline.downloader.parse_input(reel_url)
                if not shortcode:
                    continue
                if not latest_shortcode:
                    latest_shortcode = shortcode
                if self.db.is_processed(shortcode):
                    continue

                logger.info(f"Channel Radar detected new Reel for @{handle}: {canonical_url}")
                intent = f"Radar: Monitored creator @{handle}"
                try:
                    success, msg, _ = await self.pipeline.process_url(f"{canonical_url} {intent}")
                    if success:
                        total_new += 1
                except Exception as e:
                    logger.error(f"Error processing radar reel {canonical_url}: {e}")

            self.db.update_channel_last_checked(handle, last_shortcode=latest_shortcode)

        return {"scanned_channels": len(channels), "new_reels_processed": total_new}

    async def poll_instagram_dms(self) -> list[tuple[bool, str, int | None]]:
        if self.instagram_receiver:
            try:
                return await self.instagram_receiver.poll_direct_inbox()
            except Exception as e:
                logger.warning(f"Instagram DM polling encountered error: {e}")
        return []

    def start(self) -> None:
        self.scheduler.add_job(self.send_weekly_digest, "cron", day_of_week="sun", hour=9, minute=0)
        if self.canary_url:
            self.scheduler.add_job(self.run_canary_test, "cron", day_of_week="tue", hour=3, minute=0)
        if self.channel_check_interval_mins > 0:
            self.scheduler.add_job(self.scan_tracked_channels, "interval", minutes=self.channel_check_interval_mins)
        if self.instagram_receiver and getattr(self.instagram_receiver, "session_id", None):
            self.scheduler.add_job(self.poll_instagram_dms, "interval", minutes=2)
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown()
