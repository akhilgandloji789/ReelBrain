import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, User, Message, Chat, CallbackQuery
from main import (
    handle_text,
    handle_callback_query,
    handle_status,
    handle_canary,
    handle_ask,
    handle_grocery,
    handle_code,
    handle_edit,
    handle_export
)


@pytest.fixture
def mock_context():
    context = MagicMock()
    context.application.bot_data = {
        "pipeline": MagicMock(),
        "search_engine": MagicMock(),
        "actions": MagicMock(),
        "settings": MagicMock(),
        "scheduler": MagicMock()
    }
    context.args = []
    return context


@pytest.mark.asyncio
async def test_handle_text_instagram_url(mock_context):
    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    status_msg = MagicMock(spec=Message)
    status_msg.edit_text = AsyncMock()
    message.reply_text = AsyncMock(return_value=status_msg)
    message.text = "https://www.instagram.com/reel/C-12345/ Check this meal prep"
    update.message = message
    update.effective_user = MagicMock(id=123)

    pipeline = mock_context.application.bot_data["pipeline"]
    pipeline.process_url = AsyncMock(return_value=(True, "Indexed Oats under #recipe", 12))

    await handle_text(update, mock_context)

    pipeline.process_url.assert_awaited_once_with(
        "https://www.instagram.com/reel/C-12345/ Check this meal prep",
        user_id=123
    )
    status_msg.edit_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_text_question(mock_context):
    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    message.text = "What was the protein oatmeal recipe?"
    update.message = message
    update.effective_user = MagicMock(id=123)

    search_engine = mock_context.application.bot_data["search_engine"]
    search_engine.answer_conversational_query.return_value = "Use 50g oats and whey."

    await handle_text(update, mock_context)

    search_engine.answer_conversational_query.assert_called_once_with("What was the protein oatmeal recipe?")
    message.reply_text.assert_awaited_once_with("Use 50g oats and whey.", parse_mode="HTML")


@pytest.mark.asyncio
async def test_handle_callback_grocery(mock_context):
    update = MagicMock(spec=Update)
    query = MagicMock(spec=CallbackQuery)
    query.data = "grocery:42"
    query.answer = AsyncMock()
    msg = MagicMock(spec=Message)
    msg.reply_text = AsyncMock()
    query.message = msg
    update.callback_query = query

    actions = mock_context.application.bot_data["actions"]
    actions.generate_grocery_list.return_value = "• [ ] Oats"

    await handle_callback_query(update, mock_context)

    query.answer.assert_awaited_once()
    actions.generate_grocery_list.assert_called_once_with(42)
    msg.reply_text.assert_awaited_once_with("• [ ] Oats", parse_mode="HTML")


@pytest.mark.asyncio
async def test_handle_status(mock_context):
    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    update.message = message

    pipeline = mock_context.application.bot_data["pipeline"]
    pipeline.metrics.get_status.return_value = {
        "uptime_mins": 10,
        "last_latency_sec": 12.5
    }
    pipeline.db.get_metrics_summary.return_value = {
        "total_reels": 5,
        "total_entities": 20,
        "total_embeddings": 5,
        "total_actions": 3
    }

    await handle_status(update, mock_context)
    message.reply_text.assert_awaited_once()
    sent_text = message.reply_text.call_args[0][0]
    assert "ReelMind Telemetry & Status" in sent_text
    assert "Total Indexed Reels:</b> 5" in sent_text


@pytest.mark.asyncio
async def test_handle_canary(mock_context):
    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    status_msg = MagicMock(spec=Message)
    status_msg.edit_text = AsyncMock()
    message.reply_text = AsyncMock(return_value=status_msg)
    update.message = message

    scheduler = mock_context.application.bot_data["scheduler"]
    scheduler.run_canary_test = AsyncMock(return_value=(True, "Operational"))

    await handle_canary(update, mock_context)
    status_msg.edit_text.assert_awaited_once()
    assert "Passed" in status_msg.edit_text.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_edit(mock_context):
    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    update.message = message
    mock_context.args = ["5", "Updated", "ingredient", "name"]

    actions = mock_context.application.bot_data["actions"]
    actions.apply_edit.return_value = (True, "✅ Updated")

    await handle_edit(update, mock_context)
    actions.apply_edit.assert_called_once_with(5, "Updated ingredient name")
    message.reply_text.assert_awaited_once_with("✅ Updated")


@pytest.mark.asyncio
async def test_handle_export(tmp_path, mock_context):
    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    message.reply_document = AsyncMock()
    update.message = message
    mock_context.args = ["json"]

    dummy_export = tmp_path / "export.json"
    dummy_export.write_text("{}", encoding="utf-8")

    settings = mock_context.application.bot_data["settings"]
    settings.TEMP_DIR = tmp_path

    actions = mock_context.application.bot_data["actions"]
    actions.export_json.return_value = dummy_export

    await handle_export(update, mock_context)
    message.reply_document.assert_awaited_once()
