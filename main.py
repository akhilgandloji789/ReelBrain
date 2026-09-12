import argparse
import asyncio
import logging
from pathlib import Path
from telegram import Update
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
from modules.search import SearchEngine
from modules.actions import ActionHandler
from modules.scheduler import SchedulerService

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("reelmind")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
        
    text = update.message.text.strip()
    user_id = update.effective_user.id if update.effective_user else None
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    search_engine: SearchEngine = context.application.bot_data["search_engine"]

    if "instagram.com" in text:
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
    actions: ActionHandler = context.application.bot_data["actions"]

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


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    pipeline: ReelPipeline = context.application.bot_data["pipeline"]
    metrics = pipeline.metrics.get_status()
    db_metrics = pipeline.db.get_metrics_summary()

    msg = (
        f"📊 <b>ReelMind Telemetry & Status:</b>\n\n"
        f"⏱️ <b>Uptime:</b> {metrics['uptime_mins']} minutes\n"
        f"⚡ <b>Last Latency:</b> {metrics['last_latency_sec']}s\n"
        f"✅ <b>Total Indexed Reels:</b> {db_metrics['total_reels']}\n"
        f"🧬 <b>Total Grounded Entities:</b> {db_metrics['total_entities']}\n"
        f"🧠 <b>Total Embeddings:</b> {db_metrics['total_embeddings']}\n"
        f"🛠️ <b>Action Invocations:</b> {db_metrics['total_actions']}\n"
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
    settings: Settings = context.application.bot_data["settings"]
    export_dir = settings.TEMP_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        export_file = actions.export_json(export_dir / "reelmind_export.json")
    else:
        export_file = actions.export_markdown(export_dir / "reelmind_export.md")

    with open(export_file, "rb") as fp:
        await update.message.reply_document(document=fp, caption=f"📦 Exported ReelMind knowledge ({fmt.upper()})")


def run_bot(pipeline: ReelPipeline, search_engine: SearchEngine, actions: ActionHandler, settings: Settings) -> None:
    pipeline.startup_recovery()
    
    scheduler_holder: dict[str, SchedulerService | None] = {"service": None}

    async def post_init(application) -> None:
        scheduler = SchedulerService(
            db=pipeline.db,
            bot=application.bot,
            chat_id=settings.TELEGRAM_GROUP_CHAT_ID,
            pipeline=pipeline,
            canary_url=settings.CANARY_REEL_URL
        )
        scheduler.start()
        scheduler_holder["service"] = scheduler
        application.bot_data["scheduler"] = scheduler
        logger.info("Scheduler started successfully inside event loop.")

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

    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(CommandHandler("canary", handle_canary))
    app.add_handler(CommandHandler("ask", handle_ask))
    app.add_handler(CommandHandler("grocery", handle_grocery))
    app.add_handler(CommandHandler("code", handle_code))
    app.add_handler(CommandHandler("edit", handle_edit))
    app.add_handler(CommandHandler("export", handle_export))
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
