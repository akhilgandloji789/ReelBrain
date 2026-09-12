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
        canary_url: str | None = None
    ):
        self.db = db
        self.bot = bot
        self.chat_id = chat_id
        self.pipeline = pipeline
        self.canary_url = canary_url
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

    def start(self) -> None:
        self.scheduler.add_job(self.send_weekly_digest, "cron", day_of_week="sun", hour=9, minute=0)
        if self.canary_url:
            self.scheduler.add_job(self.run_canary_test, "cron", day_of_week="tue", hour=3, minute=0)
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown()
