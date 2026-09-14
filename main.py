import argparse
import asyncio
import html
import logging
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
)
from config import Settings, get_settings
from pipeline import ReelPipeline
from modules.storage import clean_handle
from modules.search import SearchEngine
from modules.actions import ActionHandler
from modules.scheduler import SchedulerService
from modules.instagram_receiver import InstagramReceiver

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("reelmind")


CATEGORY_DISPLAY_MAP = {
    "recipe": "🍳 Recipes",
    "tech": "💻 Tech & AI",
    "workout": "💪 Fitness",
    "idea": "💡 Ideas",
    "travel": "✈️ Travel",
    "finance": "💰 Finance",
    "other": "📌 Other",
}

CATEGORY_ALIASES = {
    "recipe": "recipe",
    "recipes": "recipe",
    "food": "recipe",
    "cooking": "recipe",
    "tech": "tech",
    "technology": "tech",
    "ai": "tech",
    "code": "tech",
    "coding": "tech",
    "workout": "workout",
    "fitness": "workout",
    "gym": "workout",
    "exercise": "workout",
    "idea": "idea",
    "ideas": "idea",
    "book": "idea",
    "books": "idea",
    "travel": "travel",
    "trip": "travel",
    "finance": "finance",
    "money": "finance",
    "other": "other",
    "all": "all",
}


def get_temp_disk_usage_mb(temp_dir: Path) -> float:
    if not temp_dir.exists():
        return 0.0
    total_bytes = 0
    for f in temp_dir.rglob("*"):
        try:
            if f.is_file():
                total_bytes += f.stat().st_size
        except (OSError, FileNotFoundError):
            pass
    return round(total_bytes / (1024 * 1024), 2)


def build_topic_menu_markup(db) -> tuple[str, InlineKeyboardMarkup]:
    counts = db.get_category_counts()
    total = sum(counts.values())

    rows = []
    primary = ["recipe", "tech", "workout", "idea"]
    primary_buttons = []
    for cat in primary:
        c = counts.get(cat, 0)
        label = f"{CATEGORY_DISPLAY_MAP.get(cat, cat.title())} ({c})"
        primary_buttons.append(InlineKeyboardButton(label, callback_data=f"topic_view:{cat}:0"))

    for i in range(0, len(primary_buttons), 2):
        rows.append(primary_buttons[i:i+2])

    secondary_buttons = []
    for cat, count in counts.items():
        if cat not in primary and count > 0:
            label = f"{CATEGORY_DISPLAY_MAP.get(cat, f'#{cat}')} ({count})"
            secondary_buttons.append(InlineKeyboardButton(label, callback_data=f"topic_view:{cat}:0"))

    for i in range(0, len(secondary_buttons), 2):
        rows.append(secondary_buttons[i:i+2])

    rows.append([InlineKeyboardButton(f"📋 View All ({total})", callback_data="topic_view:all:0")])

    text = (
        "📚 <b>ReelMind In-Group Topic Filter</b>\n\n"
        "Browse and filter your saved Reels by category directly in Telegram.\n"
        "Tap a topic pill below to view saved reels and notes:"
    )
    return text, InlineKeyboardMarkup(rows)


def build_category_reels_view(
    db,
    category: str,
    page: int = 0,
    page_size: int = 5
) -> tuple[str, InlineKeyboardMarkup]:
    raw_cat = category.strip().lstrip("#").lower()
    cat_key = CATEGORY_ALIASES.get(raw_cat, raw_cat)
    reels, total = db.get_reels_by_category(cat_key, limit=page_size, offset=max(0, page) * page_size)

    total_pages = max(1, (total + page_size - 1) // page_size)
    clamped_page = max(0, min(page, total_pages - 1)) if total > 0 else 0

    if total > 0 and not reels and clamped_page != page:
        page = clamped_page
        reels, total = db.get_reels_by_category(cat_key, limit=page_size, offset=page * page_size)
    else:
        page = clamped_page

    if cat_key == "all":
        topic_title = "📋 All Saved Reels"
    else:
        topic_title = CATEGORY_DISPLAY_MAP.get(cat_key, f"#{cat_key}")

    if not reels:
        text = (
            f"📂 <b>Topic: {topic_title}</b>\n\n"
            f"<i>No saved reels found in this topic yet.</i>\n\n"
            f"Share an Instagram Reel link in this group to automatically index it!"
        )
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Topics", callback_data="topic_home")]])
        return text, markup

    text_lines = [
        f"📂 <b>Topic: {topic_title}</b> (Page {page + 1}/{total_pages} — {total} items)\n"
    ]

    action_rows = []
    for idx, r in enumerate(reels, 1):
        item_num = page * page_size + idx
        safe_title = html.escape(r.get("title") or "Untitled")
        raw_tldr = r.get("raw_transcript") or ""
        safe_tldr = html.escape(raw_tldr[:120] + "..." if len(raw_tldr) > 120 else raw_tldr)
        safe_url = html.escape(r.get("url") or "#")

        text_lines.append(f"{item_num}. <b>{safe_title}</b>")
        if safe_tldr:
            text_lines.append(f"📌 <i>{safe_tldr}</i>")
        text_lines.append(f"🔗 <a href=\"{safe_url}\">Original Reel</a>\n")

        row = [InlineKeyboardButton(f"📖 #{item_num} Note", callback_data=f"view_note:{r['id']}")]
        cat_item = (r.get("category") or "").lower()
        if cat_item == "recipe":
            row.append(InlineKeyboardButton(f"🛒 #{item_num} Grocery", callback_data=f"grocery:{r['id']}"))
        elif cat_item == "tech":
            row.append(InlineKeyboardButton(f"💻 #{item_num} Code", callback_data=f"code:{r['id']}"))
        action_rows.append(row)

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀️ Prev", callback_data=f"topic_view:{cat_key}:{page - 1}"))
    if (page + 1) * page_size < total:
        nav_row.append(InlineKeyboardButton("▶️ Next", callback_data=f"topic_view:{cat_key}:{page + 1}"))

    if nav_row:
        action_rows.append(nav_row)

    action_rows.append([InlineKeyboardButton("⬅️ Back to Topics", callback_data="topic_home")])

    return "\n".join(text_lines), InlineKeyboardMarkup(action_rows)



async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
        
    text = update.message.text.strip()
    user_id = update.effective_user.id if update.effective_user else None
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    search_engine: SearchEngine = context.application.bot_data["search_engine"]

    downloader = getattr(pipeline, "downloader", None)
    is_reel = False
    if downloader and hasattr(downloader, "parse_input"):
        try:
            sc, _, _ = downloader.parse_input(text)
            is_reel = bool(sc)
        except Exception:
            is_reel = "instagram.com" in text or "instagr.am" in text
    else:
        is_reel = "instagram.com" in text or "instagr.am" in text

    if is_reel:
        status_msg = await update.message.reply_text("⏳ Processing reel... Analyzing video & audio with Gemini...")
        success, result_text, thread_id = await pipeline.process_url(text, user_id=user_id)
        await status_msg.edit_text(result_text, parse_mode="HTML")
    else:
        answer = search_engine.answer_conversational_query(text)
        await update.message.reply_text(answer, parse_mode="HTML")


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    actions: ActionHandler | None = context.application.bot_data.get("actions")
    pipeline: ReelPipeline | None = context.application.bot_data.get("pipeline")

    if not query.message:
        return

    if data.startswith("grocery:"):
        reel_id = int(data.split(":")[1])
        grocery_text = actions.generate_grocery_list(reel_id)
        await query.message.reply_text(grocery_text, parse_mode="HTML")
    elif data.startswith("code:"):
        reel_id = int(data.split(":")[1])
        code_text = actions.generate_code_block(reel_id)
        await query.message.reply_text(code_text, parse_mode="HTML")
    elif data.startswith("thumb_up:"):
        reel_id = int(data.split(":")[1])
        msg = actions.record_feedback(reel_id, "up")
        await query.message.reply_text(msg)
    elif data.startswith("thumb_down:"):
        reel_id = int(data.split(":")[1])
        msg = actions.record_feedback(reel_id, "down")
        await query.message.reply_text(msg)
    elif data.startswith("ask:"):
        await query.message.reply_text("💬 Reply with <code>/ask &lt;your question&gt;</code>", parse_mode="HTML")
    elif data.startswith("edit:"):
        await query.message.reply_text("✏️ Use <code>/edit &lt;entity_id&gt; &lt;new text&gt;</code> to correct any item.", parse_mode="HTML")
    elif data.startswith("topic_view:"):
        parts = data.split(":")
        category = parts[1]
        page = int(parts[2]) if len(parts) > 2 else 0
        text, markup = build_category_reels_view(pipeline.db, category, page=page)
        try:
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=markup, disable_web_page_preview=True)
        except Exception as e:
            if "Message is not modified" not in str(e):
                logger.warning(f"Failed to edit topic view: {e}")
    elif data == "topic_home":
        text, markup = build_topic_menu_markup(pipeline.db)
        try:
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=markup)
        except Exception as e:
            if "Message is not modified" not in str(e):
                logger.warning(f"Failed to edit topic home: {e}")
    elif data.startswith("view_note:"):
        reel_id = int(data.split(":")[1])
        note_text = actions.get_reel_summary_html(reel_id)
        reel = pipeline.db.get_reel_by_id(reel_id)
        cat = (reel.get("category") or "other").lower() if reel else "other"

        buttons = []
        act_row = []
        if cat == "recipe":
            act_row.append(InlineKeyboardButton("🛒 Grocery List", callback_data=f"grocery:{reel_id}"))
        elif cat == "tech":
            act_row.append(InlineKeyboardButton("💻 Copy Code", callback_data=f"code:{reel_id}"))
        if act_row:
            buttons.append(act_row)
        buttons.append([
            InlineKeyboardButton("👍 Accurate", callback_data=f"thumb_up:{reel_id}"),
            InlineKeyboardButton("👎 Inaccurate", callback_data=f"thumb_down:{reel_id}")
        ])
        buttons.append([InlineKeyboardButton("⬅️ Back to Topics", callback_data="topic_home")])
        await query.message.reply_text(note_text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons), disable_web_page_preview=True)


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    settings: Settings = context.application.bot_data.get("settings")
    metrics = pipeline.metrics.get_status()
    db_metrics = pipeline.db.get_metrics_summary()

    temp_usage = get_temp_disk_usage_mb(settings.TEMP_DIR) if settings else 0.0
    ephemeral_mode = "Active (Zero-Disk steady-state)" if (settings and getattr(settings, "CLEANUP_TEMP", True)) else "Disabled"
    tracked_count = len(pipeline.db.get_tracked_channels(active_only=True))

    msg = (
        f"📊 <b>ReelMind Telemetry & Status:</b>\n\n"
        f"⏱️ <b>Uptime:</b> {metrics['uptime_mins']} minutes\n"
        f"⚡ <b>Last Latency:</b> {metrics['last_latency_sec']}s\n"
        f"✅ <b>Total Indexed Reels:</b> {db_metrics['total_reels']}\n"
        f"🧬 <b>Total Grounded Entities:</b> {db_metrics['total_entities']}\n"
        f"🧠 <b>Total Embeddings:</b> {db_metrics['total_embeddings']}\n"
        f"🛠️ <b>Action Invocations:</b> {db_metrics['total_actions']}\n"
        f"📡 <b>Tracked Creators:</b> {tracked_count}\n"
        f"💾 <b>Temp Disk Usage:</b> {temp_usage} MB (Ephemeral: {ephemeral_mode})\n"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def handle_canary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    scheduler: SchedulerService = context.application.bot_data["scheduler"]
    status_msg = await update.message.reply_text("🧪 Running live Canary health diagnostic...")
    ok, details = await scheduler.run_canary_test()
    if ok:
        await status_msg.edit_text("✅ <b>Canary Check Passed!</b>\nTelegram, SQLite, Gemini, and Downloader are operational.")
    else:
        await status_msg.edit_text(f"🚨 <b>Canary Check Failed!</b>\n{details}")


async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: <code>/ask what was the recipe with oats?</code>", parse_mode="HTML")
        return
    query = " ".join(context.args)
    search_engine: SearchEngine = context.application.bot_data["search_engine"]
    answer = search_engine.answer_conversational_query(query)
    await update.message.reply_text(answer, parse_mode="HTML")


async def handle_grocery(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    actions: ActionHandler = context.application.bot_data["actions"]
    if context.args:
        try:
            reel_id = int(context.args[0])
            msg = actions.generate_grocery_list(reel_id)
            await update.message.reply_text(msg, parse_mode="HTML")
            return
        except ValueError:
            pass
    await update.message.reply_text("Usage: <code>/grocery &lt;reel_id&gt;</code>", parse_mode="HTML")


async def handle_code(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    actions: ActionHandler = context.application.bot_data["actions"]
    if context.args:
        try:
            reel_id = int(context.args[0])
            msg = actions.generate_code_block(reel_id)
            await update.message.reply_text(msg, parse_mode="HTML")
            return
        except ValueError:
            pass
    await update.message.reply_text("Usage: <code>/code &lt;reel_id&gt;</code>", parse_mode="HTML")


async def handle_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    actions: ActionHandler = context.application.bot_data["actions"]
    if context.args and len(context.args) >= 2:
        try:
            entity_id = int(context.args[0])
            new_text = " ".join(context.args[1:])
            ok, msg = actions.apply_edit(entity_id, new_text)
            await update.message.reply_text(msg)
            return
        except ValueError:
            pass
    await update.message.reply_text("Usage: <code>/edit &lt;entity_id&gt; &lt;corrected text&gt;</code>", parse_mode="HTML")


async def handle_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    fmt = context.args[0].lower() if context.args else "md"
    actions: ActionHandler = context.application.bot_data["actions"]
    settings: Settings = context.application.bot_data.get("settings")
    export_dir = settings.TEMP_DIR / "exports" if settings else Path("temp/exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        export_file = actions.export_json(export_dir / "reelmind_export.json")
    else:
        export_file = actions.export_markdown(export_dir / "reelmind_export.md")

    try:
        with open(export_file, "rb") as fp:
            await update.message.reply_document(document=fp, caption=f"📦 Exported ReelMind knowledge ({fmt.upper()})")
    finally:
        if settings and getattr(settings, "CLEANUP_TEMP", True) and export_file.exists():
            try:
                export_file.unlink()
            except Exception:
                pass


async def handle_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    if context.args:
        category_query = " ".join(context.args).strip().lstrip("#").lower()
        text, markup = build_category_reels_view(pipeline.db, category_query, page=0)
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=markup, disable_web_page_preview=True)
    else:
        text, markup = build_topic_menu_markup(pipeline.db)
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=markup)


async def handle_track(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: <code>/track @handle</code>", parse_mode="HTML")
        return
    handle = clean_handle(context.args[0])
    if not handle:
        await update.message.reply_text("⚠️ Please provide a valid Instagram handle.", parse_mode="HTML")
        return
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    pipeline.db.add_tracked_channel(handle)
    await update.message.reply_text(
        f"✅ Now monitoring <b>@{html.escape(handle)}</b>!\nReelMind Radar will automatically index new Reels.",
        parse_mode="HTML"
    )


async def handle_untrack(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: <code>/untrack @handle</code>", parse_mode="HTML")
        return
    handle = clean_handle(context.args[0])
    if not handle:
        await update.message.reply_text("⚠️ Please provide a valid Instagram handle to untrack.", parse_mode="HTML")
        return
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    ok = pipeline.db.remove_tracked_channel(handle)
    if ok:
        await update.message.reply_text(f"🗑️ Stopped monitoring <b>@{html.escape(handle)}</b>.", parse_mode="HTML")
    else:
        await update.message.reply_text(f"⚠️ Creator <b>@{html.escape(handle)}</b> was not found in tracked channels.", parse_mode="HTML")


async def handle_tracked(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    channels = pipeline.db.get_tracked_channels(active_only=True)
    if not channels:
        await update.message.reply_text(
            "📡 No creators currently monitored.\nUse <code>/track @handle</code> to start monitoring creators.",
            parse_mode="HTML"
        )
        return

    lines = [f"📡 <b>Monitored Creators ({len(channels)}):</b>\n"]
    for ch in channels:
        h = html.escape(ch["handle"])
        last_chk = ch.get("last_checked_at")
        if last_chk:
            last_str = last_chk.replace("T", " ")[:19] + " UTC"
        else:
            last_str = "Never"
        lines.append(f"• <b>@{h}</b> (Last scan: <i>{last_str}</i>)")

    lines.append("\n<i>Trigger immediate check with /check_now</i>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def handle_check_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    scheduler: SchedulerService = context.application.bot_data.get("scheduler")
    if not scheduler:
        await update.message.reply_text("⚠️ Scheduler service not running.", parse_mode="HTML")
        return

    status_msg = await update.message.reply_text("📡 Scanning all tracked creators for new Reels...")
    try:
        res = await scheduler.scan_tracked_channels()
        await status_msg.edit_text(
            f"📡 <b>Radar Scan Complete:</b>\nScanned <b>{res['scanned_channels']}</b> creators, indexed <b>{res['new_reels_processed']}</b> new Reels.",
            parse_mode="HTML"
        )
    except Exception as e:
        await status_msg.edit_text(f"⚠️ Radar scan encountered error: {html.escape(str(e))}", parse_mode="HTML")


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    msg = (
        "🧠 <b>ReelMind / ReelBrain — AI Second Brain Commands</b>\n\n"
        "Here are all the available features you can use:\n\n"
        "📂 <b>Search & Browse:</b>\n"
        "• <code>/filter</code> or <code>/topics</code> — Interactive topic buttons (#recipe, #tech, #fitness, #ideas)\n"
        "• <code>/ask &lt;question&gt;</code> — Semantic search with exact video timestamps (e.g. <i>/ask how to win a hackathon?</i>)\n"
        "• <code>/reels &lt;topic&gt;</code> — Quick filter by category (e.g. <i>/reels tech</i>)\n\n"
        "📡 <b>Creator Radar (Auto-Monitoring):</b>\n"
        "• <code>/track @handle</code> — Auto-monitor an Instagram creator for new reels\n"
        "• <code>/untrack @handle</code> — Stop monitoring a creator\n"
        "• <code>/tracked</code> — List all active radar channels & check times\n"
        "• <code>/check_now</code> — Force an immediate scan for new reels right now\n\n"
        "⚡ <b>Action Tools:</b>\n"
        "• <code>/grocery &lt;reel_id&gt;</code> — Generate clean shopping checklist\n"
        "• <code>/code &lt;reel_id&gt;</code> — Extract syntax-highlighted code snippets\n"
        "• <code>/edit &lt;entity_id&gt; &lt;new text&gt;</code> — Correct an entity & re-embed\n"
        "• <code>/export md</code> or <code>/export json</code> — Download entire database\n\n"
        "📊 <b>System & Health:</b>\n"
        "• <code>/status</code> — Telemetry, total reels & disk footprint\n"
        "• <code>/canary</code> — Run end-to-end diagnostic test\n\n"
        "💡 <i>Tip: You can also DM any Reel to your receiver Instagram account or paste the link here!</i>"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


def run_bot(pipeline: ReelPipeline, search_engine: SearchEngine, actions: ActionHandler, settings: Settings) -> None:
    pipeline.startup_recovery()

    instagram_receiver = InstagramReceiver(
        pipeline=pipeline,
        db=pipeline.db,
        session_id=settings.INSTAGRAM_SESSION_ID,
        verify_token=settings.INSTAGRAM_WEBHOOK_VERIFY_TOKEN,
        downloader=pipeline.downloader
    )

    scheduler_holder: dict[str, SchedulerService | None] = {"service": None}

    async def post_init(application) -> None:
        scheduler = SchedulerService(
            db=pipeline.db,
            bot=application.bot,
            chat_id=settings.TELEGRAM_GROUP_CHAT_ID,
            pipeline=pipeline,
            canary_url=settings.CANARY_REEL_URL,
            channel_check_interval_mins=settings.CHANNEL_CHECK_INTERVAL_MINS,
            instagram_receiver=instagram_receiver
        )
        scheduler.start()
        scheduler_holder["service"] = scheduler
        application.bot_data["scheduler"] = scheduler
        application.bot_data["instagram_receiver"] = instagram_receiver
        logger.info("Scheduler started successfully inside event loop.")

        # Register bot commands with Telegram so typing '/' shows autocomplete menu
        commands = [
            BotCommand("filter", "Browse & filter saved reels by topic"),
            BotCommand("topics", "Interactive topic menu"),
            BotCommand("ask", "Search your second brain with AI timestamps"),
            BotCommand("track", "Auto-monitor an Instagram creator (@handle)"),
            BotCommand("untrack", "Stop monitoring an Instagram creator"),
            BotCommand("tracked", "View all active creator radar channels"),
            BotCommand("check_now", "Trigger instant radar scan for new reels"),
            BotCommand("status", "View system telemetry & disk footprint"),
            BotCommand("canary", "Run end-to-end health diagnostic"),
            BotCommand("grocery", "Extract grocery checklist from a recipe"),
            BotCommand("code", "Extract clean code blocks from a tech reel"),
            BotCommand("edit", "Correct an entity text & refresh embedding"),
            BotCommand("export", "Export knowledge base as Markdown or JSON"),
            BotCommand("help", "Show all features and usage guide"),
            BotCommand("start", "Welcome message & getting started"),
        ]
        try:
            await application.bot.set_my_commands(commands)
            logger.info("Registered bot commands with Telegram for '/' popup menu.")
        except Exception as e:
            logger.warning(f"Could not register bot commands: {e}")

    async def post_shutdown(application) -> None:
        if scheduler_holder["service"]:
            scheduler_holder["service"].shutdown()
            logger.info("Scheduler stopped cleanly.")

    app = (
        ApplicationBuilder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.bot_data["pipeline"] = pipeline
    app.bot_data["search_engine"] = search_engine
    app.bot_data["actions"] = actions
    app.bot_data["settings"] = settings
    app.bot_data["instagram_receiver"] = instagram_receiver

    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("start", handle_help))
    app.add_handler(CommandHandler("commands", handle_help))
    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(CommandHandler("canary", handle_canary))
    app.add_handler(CommandHandler("ask", handle_ask))
    app.add_handler(CommandHandler("grocery", handle_grocery))
    app.add_handler(CommandHandler("code", handle_code))
    app.add_handler(CommandHandler("edit", handle_edit))
    app.add_handler(CommandHandler("export", handle_export))
    app.add_handler(CommandHandler("filter", handle_filter))
    app.add_handler(CommandHandler("topics", handle_filter))
    app.add_handler(CommandHandler("reels", handle_filter))
    app.add_handler(CommandHandler("track", handle_track))
    app.add_handler(CommandHandler("untrack", handle_untrack))
    app.add_handler(CommandHandler("tracked", handle_tracked))
    app.add_handler(CommandHandler("check_now", handle_check_now))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    logger.info("ReelMind Bot is running! Share a Reel URL, ask a question, or use inline buttons.")
    app.run_polling()


def main() -> None:
    parser = argparse.ArgumentParser(description="ReelMind — Personal AI Second Brain")
    parser.add_argument("--url", type=str, help="Process a single Instagram Reel URL immediately via CLI")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = ReelPipeline(settings)
    pipeline.startup_recovery()
    search_engine = SearchEngine(pipeline.db, pipeline.analyzer)
    actions = ActionHandler(pipeline.db, pipeline.analyzer)

    if args.url:
        logger.info(f"Processing URL via CLI: {args.url}")
        success, result, _ = asyncio.run(pipeline.process_url(args.url))
        print(f"[{'SUCCESS' if success else 'FAILED'}] {result}")
    else:
        run_bot(pipeline, search_engine, actions, settings)


if __name__ == "__main__":
    main()
