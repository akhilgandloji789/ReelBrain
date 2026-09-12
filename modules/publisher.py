import html
from pathlib import Path
from typing import Any
from telegram import Bot, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from modules.analyzer import ReelAnalysisOutput


def format_timestamp(seconds: float) -> str:
    total_seconds = int(round(seconds))
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m:02d}:{s:02d}"


class TelegramPublisher:
    def __init__(
        self,
        bot_token: str,
        group_chat_id: str | int,
        topic_map: dict[str, int | None] | None = None,
        bot: Any = None
    ):
        self.bot = bot or Bot(token=bot_token)
        self.group_chat_id = group_chat_id
        self.topic_map = topic_map or {}

    def get_thread_id(self, category: str) -> int | None:
        return self.topic_map.get(category.lower())

    def format_note_html(self, analysis: ReelAnalysisOutput, original_url: str, user_intent: str | None = None) -> str:
        safe_title = html.escape(analysis.title)
        safe_tldr = html.escape(analysis.tldr)

        intent_block = ""
        if user_intent:
            intent_block = f"💡 <b>Why You Saved This:</b> {html.escape(user_intent)}\n\n"

        entity_lines = []
        for e in analysis.entities:
            ts_str = format_timestamp(e.start_ts)
            conf_str = f"{int(e.confidence * 100)}% conf" if e.confidence is not None else "verified in video"
            entity_lines.append(f"• {html.escape(e.text)} <i>({ts_str} | {conf_str})</i>")

        body_block = "\n".join(entity_lines)
        safe_url = html.escape(original_url)

        return (
            f"🎬 <b>{safe_title}</b>\n\n"
            f"{intent_block}"
            f"📌 <b>TL;DR:</b>\n{safe_tldr}\n\n"
            f"⚡ <b>Key Evidence & Steps:</b>\n{body_block}\n\n"
            f"🏷️ <i>#{analysis.category}</i>\n\n"
            f'🔗 <b>Original Reel:</b> <a href="{safe_url}">{safe_url}</a>'
        )

    def build_inline_keyboard(self, reel_id: int, category: str) -> InlineKeyboardMarkup:
        rows = []
        if category == "recipe":
            rows.append([InlineKeyboardButton("🛒 Get Grocery List", callback_data=f"grocery:{reel_id}")])
        elif category == "tech":
            rows.append([InlineKeyboardButton("💻 Copy Code", callback_data=f"code:{reel_id}")])
        
        rows.append([
            InlineKeyboardButton("🔍 Ask", callback_data=f"ask:{reel_id}"),
            InlineKeyboardButton("✏️ Edit", callback_data=f"edit:{reel_id}")
        ])
        rows.append([
            InlineKeyboardButton("👍 Accurate", callback_data=f"thumb_up:{reel_id}"),
            InlineKeyboardButton("👎 Inaccurate", callback_data=f"thumb_down:{reel_id}")
        ])
        return InlineKeyboardMarkup(rows)

    async def publish_reel(
        self,
        reel_id: int,
        analysis: ReelAnalysisOutput,
        frame_paths: list[Path | str],
        original_url: str,
        user_intent: str | None = None
    ) -> int | None:
        thread_id = self.get_thread_id(analysis.category)
        
        valid_frames = [Path(p) for p in frame_paths if Path(p).exists()]
        if valid_frames:
            media = []
            files_to_close = []
            try:
                for frame in valid_frames[:4]:
                    fp = open(frame, "rb")
                    files_to_close.append(fp)
                    media.append(InputMediaPhoto(media=fp))
                
                await self.bot.send_media_group(
                    chat_id=self.group_chat_id,
                    message_thread_id=thread_id,
                    media=media
                )
            finally:
                for fp in files_to_close:
                    fp.close()

        text_content = self.format_note_html(analysis, original_url, user_intent)
        reply_markup = self.build_inline_keyboard(reel_id, analysis.category)
        
        await self.bot.send_message(
            chat_id=self.group_chat_id,
            message_thread_id=thread_id,
            text=text_content,
            parse_mode="HTML",
            reply_markup=reply_markup,
            disable_web_page_preview=False
        )
        return thread_id
