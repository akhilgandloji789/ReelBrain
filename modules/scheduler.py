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
        recent = self.db.get_recent_reels(days=7)
        if not recent:
            return None

        recipes = [r for r in recent if r.get("category") == "recipe"]
        workouts = [r for r in recent if r.get("category") == "workout"]
        tech = [r for r in recent if r.get("category") == "tech"]
        others = [r for r in recent if r.get("category") not in ("recipe", "workout", "tech")]

        lines = ["🧠 <b>YOUR REELMIND — SUNDAY REVIEW</b>\n"]
        if recipes:
            lines.append(f"🍳 <b>{len(recipes)} recipe{'s' if len(recipes) > 1 else ''} saved:</b>")
            for r in recipes[:3]:
                title = r.get("title") or "Recipe"
                url = r.get("url") or "#"
                lines.append(f"• <a href='{url}'>{title}</a>")
            lines.append("")
        if workouts:
            lines.append(f"🏋️ <b>{len(workouts)} workout{'s' if len(workouts) > 1 else ''} saved:</b>")
            for w in workouts[:3]:
                title = w.get("title") or "Workout"
                url = w.get("url") or "#"
                lines.append(f"• <a href='{url}'>{title}</a>")
            lines.append("")
        if tech:
            lines.append(f"💻 <b>{len(tech)} tech idea{'s' if len(tech) > 1 else ''} saved:</b>")
            for t in tech[:3]:
                title = t.get("title") or "Tech Item"
                url = t.get("url") or "#"
                lines.append(f"• <a href='{url}'>{title}</a>")
            lines.append("")
        if others:
            lines.append(f"💡 <b>{len(others)} other discovery item{'s' if len(others) > 1 else ''} saved.</b>\n")

        lines.append("<i>Ask me anything about these with /ask!</i>")
        return "\n".join(lines)

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
