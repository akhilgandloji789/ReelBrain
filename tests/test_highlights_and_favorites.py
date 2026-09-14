import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from modules.storage import ReelDatabase
from modules.scheduler import SchedulerService
from main import (
    build_highlights_view,
    build_favorites_view,
    handle_highlights,
    handle_favorites,
    handle_favorite_cmd,
    handle_digest,
    handle_wishlist,
)


@pytest.fixture
def test_db(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    r1 = db.add_reel("https://www.instagram.com/reel/C-pasta1/", "C-pasta1", "recipe", "Creamy Garlic Pasta", raw_transcript="Cook pasta in garlic butter")
    db.update_reel_status(r1, "COMPLETED")
    r2 = db.add_reel("https://www.instagram.com/reel/C-ai2/", "C-ai2", "tech", "Build AI Agent with Python", raw_transcript="Use LangChain and FastAPI")
    db.update_reel_status(r2, "COMPLETED")
    return db


def test_build_highlights_view(test_db: ReelDatabase):
    text, markup = build_highlights_view(test_db, period="today")
    assert "ReelBrain Highlights — 📅 Today" in text
    assert "2 Total Reels Indexed" in text
    assert "Creamy Garlic Pasta" in text
    assert "Build AI Agent with Python" in text

    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "• Today •" in buttons
    assert "🗓️ Week" in buttons
    assert "🗓️ Month" in buttons
    assert "⬅️ Back to Topics" in buttons


def test_build_favorites_view(test_db: ReelDatabase):
    empty_text, empty_markup = build_favorites_view(test_db, page=0)
    assert "You haven't favorited any reels yet" in empty_text

    test_db.toggle_favorite(1)
    text, markup = build_favorites_view(test_db, page=0)
    assert "Your Favorited Reels" in text
    assert "1 items" in text
    assert "Creamy Garlic Pasta" in text

    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "📖 #1 Note" in buttons
    assert "❌ Unstar" in buttons


def test_sunday_master_digest(test_db: ReelDatabase):
    bot_mock = MagicMock()
    scheduler = SchedulerService(db=test_db, bot=bot_mock, chat_id="12345")

    digest = scheduler.build_weekly_digest()
    assert digest is not None
    assert "REELBRAIN — SUNDAY MASTER DIGEST" in digest
    assert "Weekly Executive Review: 2 Reels Indexed" in digest
    assert "🍳 Recipes (<b>1</b>):" in digest
    assert "Creamy Garlic Pasta" in digest
    assert "💻 Tech & AI (<b>1</b>):" in digest
    assert "Build AI Agent with Python" in digest


@pytest.mark.asyncio
async def test_handle_highlights_command(test_db: ReelDatabase):
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.args = ["week"]
    pipeline = MagicMock()
    pipeline.db = test_db
    context.application.bot_data = {"pipeline": pipeline}

    await handle_highlights(update, context)
    update.message.reply_text.assert_called_once()
    args, kwargs = update.message.reply_text.call_args
    assert "ReelBrain Highlights — 🗓️ This Week" in args[0]


@pytest.mark.asyncio
async def test_handle_favorites_command(test_db: ReelDatabase):
    test_db.toggle_favorite(1)
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    pipeline = MagicMock()
    pipeline.db = test_db
    context.application.bot_data = {"pipeline": pipeline}

    await handle_favorites(update, context)
    update.message.reply_text.assert_called_once()
    args, kwargs = update.message.reply_text.call_args
    assert "Your Favorited Reels" in args[0]
    assert "Creamy Garlic Pasta" in args[0]


@pytest.mark.asyncio
async def test_handle_favorite_cmd_toggle(test_db: ReelDatabase):
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = "/favorite 1"
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.args = ["1"]
    pipeline = MagicMock()
    pipeline.db = test_db
    context.application.bot_data = {"pipeline": pipeline}

    await handle_favorite_cmd(update, context)
    assert test_db.is_favorite(1) is True
    assert "added to Favorites" in update.message.reply_text.call_args[0][0]

    await handle_favorite_cmd(update, context)
    assert test_db.is_favorite(1) is False
    assert "removed from Favorites" in update.message.reply_text.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_digest_command(test_db: ReelDatabase):
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    scheduler = SchedulerService(db=test_db, bot=MagicMock(), chat_id="12345")
    context = MagicMock()
    context.application.bot_data = {"scheduler": scheduler}

    await handle_digest(update, context)
    update.message.reply_text.assert_called_once()
    assert "REELBRAIN — SUNDAY MASTER DIGEST" in update.message.reply_text.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_wishlist_command(test_db: ReelDatabase):
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    pipeline = MagicMock()
    pipeline.db = test_db
    context = MagicMock()
    context.args = []
    context.application.bot_data = {"pipeline": pipeline}

    await handle_wishlist(update, context)
    update.message.reply_text.assert_called_once()
    assert "No creators currently monitored" in update.message.reply_text.call_args[0][0]
