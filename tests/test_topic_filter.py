import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from modules.storage import ReelDatabase
from telegram import Update, Message, CallbackQuery
from main import (
    build_topic_menu_markup,
    build_category_reels_view,
    handle_filter,
    handle_callback_query
)


@pytest.fixture
def populated_db(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_topic_filter.db")
    # Add recipe reels
    r1 = db.add_reel("https://www.instagram.com/reel/C-rec1/", "C-rec1", "recipe", "Avocado Toast", raw_transcript="Toast avocado with eggs")
    db.update_reel_status(r1, "COMPLETED")
    db.add_entities(r1, [{"entity_type": "ingredient", "text": "2 slices sourdough bread", "start_ts": 5.0}])

    r2 = db.add_reel("https://www.instagram.com/reel/C-rec2/", "C-rec2", "recipe", "Protein Smoothie", raw_transcript="Whey and almond milk")
    db.update_reel_status(r2, "COMPLETED")

    # Add tech reels
    r3 = db.add_reel("https://www.instagram.com/reel/C-tech1/", "C-tech1", "tech", "Docker Multi-stage", raw_transcript="Build lean containers")
    db.update_reel_status(r3, "COMPLETED")
    db.add_entities(r3, [{"entity_type": "code", "text": "FROM alpine:latest", "start_ts": 12.0}])

    # Add workout reel
    r4 = db.add_reel("https://www.instagram.com/reel/C-fit1/", "C-fit1", "workout", "Bench Press Form", raw_transcript="Retract scapula and arch")
    db.update_reel_status(r4, "COMPLETED")

    return db


def test_category_counts_and_query(populated_db: ReelDatabase):
    counts = populated_db.get_category_counts()
    assert counts.get("recipe") == 2
    assert counts.get("tech") == 1
    assert counts.get("workout") == 1
    assert counts.get("idea", 0) == 0

    # Query category
    recipe_reels, total = populated_db.get_reels_by_category("recipe", limit=10, offset=0)
    assert total == 2
    assert len(recipe_reels) == 2

    # Query all
    all_reels, total_all = populated_db.get_reels_by_category("all", limit=10, offset=0)
    assert total_all == 4
    assert len(all_reels) == 4


def test_build_topic_menu_markup(populated_db: ReelDatabase):
    text, markup = build_topic_menu_markup(populated_db)
    assert "ReelMind In-Group Topic Filter" in text

    # Extract button labels and callback datas
    buttons = [btn for row in markup.inline_keyboard for btn in row]
    btn_dict = {b.text: b.callback_data for b in buttons}

    assert any("Recipes (2)" in text for text in btn_dict.keys())
    assert any("Tech & AI (1)" in text for text in btn_dict.keys())
    assert any("Fitness (1)" in text for text in btn_dict.keys())
    assert any("View All (4)" in text for text in btn_dict.keys())


def test_build_category_reels_view(populated_db: ReelDatabase):
    # Test specific category
    text, markup = build_category_reels_view(populated_db, "recipe", page=0, page_size=1)
    assert "Topic: 🍳 Recipes" in text
    assert "Page 1/2 — 2 items" in text
    assert "Avocado Toast" in text or "Protein Smoothie" in text

    buttons = [btn for row in markup.inline_keyboard for btn in row]
    btn_texts = [b.text for b in buttons]

    assert any("Note" in b for b in btn_texts)
    assert any("Grocery" in b for b in btn_texts)
    assert "▶️ Next" in btn_texts
    assert "⬅️ Back to Topics" in btn_texts

    # Test category alias (e.g. "code" -> "tech")
    tech_text, tech_markup = build_category_reels_view(populated_db, "code", page=0)
    assert "Topic: 💻 Tech & AI" in tech_text
    assert "Docker Multi-stage" in tech_text

    tech_buttons = [b.text for row in tech_markup.inline_keyboard for b in row]
    assert any("Code" in b for b in tech_buttons)

    # Test empty category
    empty_text, empty_markup = build_category_reels_view(populated_db, "idea", page=0)
    assert "No saved reels found in this topic yet" in empty_text


@pytest.mark.asyncio
async def test_telegram_handle_filter_commands(populated_db: ReelDatabase):
    context = MagicMock()
    pipeline = MagicMock()
    pipeline.db = populated_db
    context.application.bot_data = {"pipeline": pipeline}

    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    update.message = message

    # /filter without args (shows topic menu)
    context.args = []
    await handle_filter(update, context)
    message.reply_text.assert_awaited_once()
    assert "ReelMind In-Group Topic Filter" in message.reply_text.call_args[0][0]

    # /filter tech with args
    message.reply_text.reset_mock()
    context.args = ["tech"]
    await handle_filter(update, context)
    assert "Topic: 💻 Tech & AI" in message.reply_text.call_args[0][0]
    assert "Docker Multi-stage" in message.reply_text.call_args[0][0]


@pytest.mark.asyncio
async def test_telegram_topic_callback_handling(populated_db: ReelDatabase):
    context = MagicMock()
    pipeline = MagicMock()
    pipeline.db = populated_db
    actions = MagicMock()
    actions.get_reel_summary_html.return_value = "🎬 <b>Avocado Toast Full Note</b>\n\n📌 <b>TL;DR:</b> Good breakfast"
    context.application.bot_data = {
        "pipeline": pipeline,
        "actions": actions
    }

    update = MagicMock(spec=Update)
    query = MagicMock(spec=CallbackQuery)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query_msg = MagicMock(spec=Message)
    query_msg.reply_text = AsyncMock()
    query.message = query_msg
    update.callback_query = query

    # 1. Click topic pill
    query.data = "topic_view:recipe:0"
    await handle_callback_query(update, context)
    query.answer.assert_awaited_once()
    query.edit_message_text.assert_awaited_once()
    assert "Topic: 🍳 Recipes" in query.edit_message_text.call_args[0][0]

    # 2. Click back to topics home
    query.edit_message_text.reset_mock()
    query.data = "topic_home"
    await handle_callback_query(update, context)
    assert "ReelMind In-Group Topic Filter" in query.edit_message_text.call_args[0][0]

    # 3. View full note
    query.data = "view_note:1"
    await handle_callback_query(update, context)
    actions.get_reel_summary_html.assert_called_once_with(1)
    query_msg.reply_text.assert_awaited_once()
    assert "Avocado Toast Full Note" in query_msg.reply_text.call_args[0][0]


@pytest.mark.asyncio
async def test_filter_with_hashtag_prefix(populated_db: ReelDatabase):
    context = MagicMock()
    pipeline = MagicMock()
    pipeline.db = populated_db
    context.application.bot_data = {"pipeline": pipeline}

    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    update.message = message

    # /filter #recipes
    context.args = ["#recipes"]
    await handle_filter(update, context)
    assert "Topic: 🍳 Recipes" in message.reply_text.call_args[0][0]

    # /filter #tech
    message.reply_text.reset_mock()
    context.args = ["#tech"]
    await handle_filter(update, context)
    assert "Topic: 💻 Tech & AI" in message.reply_text.call_args[0][0]


def test_pagination_boundary_clamping(populated_db: ReelDatabase):
    # Requesting out-of-bounds page should clamp to last valid page
    text, markup = build_category_reels_view(populated_db, "recipe", page=99, page_size=1)
    assert "Page 2/2" in text
    assert "Protein Smoothie" in text or "Avocado Toast" in text


@pytest.mark.asyncio
async def test_callback_query_ignores_not_modified(populated_db: ReelDatabase):
    context = MagicMock()
    pipeline = MagicMock()
    pipeline.db = populated_db
    context.application.bot_data = {"pipeline": pipeline}

    update = MagicMock(spec=Update)
    query = MagicMock(spec=CallbackQuery)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock(side_effect=RuntimeError("Message is not modified: specified new message content is identical"))
    update.callback_query = query

    query.data = "topic_view:recipe:0"
    # Should not raise exception
    await handle_callback_query(update, context)
    query.answer.assert_awaited_once()
