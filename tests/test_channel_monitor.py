import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from modules.storage import ReelDatabase
from modules.downloader import Downloader
from modules.scheduler import SchedulerService
from telegram import Update, Message
from main import handle_track, handle_untrack, handle_tracked, handle_check_now


@pytest.fixture
def temp_db(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_channel_monitor.db")
    return db


def test_tracked_channels_crud(temp_db: ReelDatabase):
    # Add creator
    ok = temp_db.add_tracked_channel("@hubermanlab")
    assert ok is True

    # Retrieve creator
    ch = temp_db.get_tracked_channel("hubermanlab")
    assert ch is not None
    assert ch["handle"] == "hubermanlab"
    assert ch["is_active"] == 1
    assert ch["last_checked_at"] is None

    # Update check time and shortcode
    temp_db.update_channel_last_checked("hubermanlab", last_shortcode="C-new123")
    ch_updated = temp_db.get_tracked_channel("hubermanlab")
    assert ch_updated["last_checked_at"] is not None
    assert ch_updated["last_shortcode"] == "C-new123"

    # Add second creator
    temp_db.add_tracked_channel("fireship_dev")
    channels = temp_db.get_tracked_channels(active_only=True)
    assert len(channels) == 2
    handles = [c["handle"] for c in channels]
    assert "hubermanlab" in handles
    assert "fireship_dev" in handles

    # Remove creator
    removed = temp_db.remove_tracked_channel("@hubermanlab")
    assert removed is True
    active_channels = temp_db.get_tracked_channels(active_only=True)
    assert len(active_channels) == 1
    assert active_channels[0]["handle"] == "fireship_dev"

    # Remove non-existent returns False
    assert temp_db.remove_tracked_channel("unknown_handle") is False


def test_downloader_get_channel_reels_mocked():
    downloader = Downloader()
    mock_info = {
        "entries": [
            {"id": "C-reel1", "url": "https://www.instagram.com/reel/C-reel1/"},
            {"id": "C-reel2", "url": None},  # should be reconstructed from id
        ]
    }
    with patch("modules.downloader.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        reels = downloader.get_channel_reels("@hubermanlab", limit=5)
        assert len(reels) == 2
        assert reels[0] == "https://www.instagram.com/reel/C-reel1/"
        assert reels[1] == "https://www.instagram.com/reel/C-reel2/"


@pytest.mark.asyncio
async def test_scheduler_scan_tracked_channels(temp_db: ReelDatabase):
    temp_db.add_tracked_channel("hubermanlab")
    
    mock_pipeline = MagicMock()
    mock_pipeline.downloader = MagicMock()
    mock_pipeline.downloader.get_channel_reels.return_value = [
        "https://www.instagram.com/reel/C-newReel1/",
        "https://www.instagram.com/reel/C-newReel2/",
    ]
    mock_pipeline.downloader.parse_input.side_effect = lambda u: (
        u.split("/reel/")[1].strip("/"),
        u,
        None
    )
    mock_pipeline.process_url = AsyncMock(return_value=(True, "Indexed", 101))

    mock_bot = MagicMock()
    service = SchedulerService(
        db=temp_db,
        bot=mock_bot,
        chat_id="-1001",
        pipeline=mock_pipeline
    )

    # First scan: processes both reels
    res = await service.scan_tracked_channels()
    assert res["scanned_channels"] == 1
    assert res["new_reels_processed"] == 2
    assert mock_pipeline.process_url.await_count == 2

    ch = temp_db.get_tracked_channel("hubermanlab")
    assert ch["last_shortcode"] == "C-newReel1"
    assert ch["last_checked_at"] is not None

    # Mark the first reel as processed in DB
    temp_db.add_reel("https://www.instagram.com/reel/C-newReel1/", "C-newReel1", "tech", "Title")
    temp_db.update_reel_status(1, "COMPLETED")

    # Second scan: should only process C-newReel2
    mock_pipeline.process_url.reset_mock()
    res2 = await service.scan_tracked_channels()
    assert res2["new_reels_processed"] == 1
    assert mock_pipeline.process_url.await_count == 1


@pytest.mark.asyncio
async def test_telegram_commands_track_and_untrack(temp_db: ReelDatabase):
    context = MagicMock()
    pipeline = MagicMock()
    pipeline.db = temp_db
    context.application.bot_data = {"pipeline": pipeline}

    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    update.message = message

    # /track without args
    context.args = []
    await handle_track(update, context)
    message.reply_text.assert_awaited()
    assert "Usage" in message.reply_text.call_args[0][0]

    # /track @hubermanlab
    message.reply_text.reset_mock()
    context.args = ["@hubermanlab"]
    await handle_track(update, context)
    assert "Now monitoring" in message.reply_text.call_args[0][0]
    assert "@hubermanlab" in message.reply_text.call_args[0][0]
    assert temp_db.get_tracked_channel("hubermanlab") is not None

    # /tracked
    message.reply_text.reset_mock()
    context.args = []
    await handle_tracked(update, context)
    assert "@hubermanlab" in message.reply_text.call_args[0][0]

    # /untrack @hubermanlab
    message.reply_text.reset_mock()
    context.args = ["@hubermanlab"]
    await handle_untrack(update, context)
    assert "Stopped monitoring" in message.reply_text.call_args[0][0]
    assert temp_db.get_tracked_channels(active_only=True) == []


@pytest.mark.asyncio
async def test_telegram_command_check_now():
    context = MagicMock()
    scheduler = MagicMock()
    scheduler.scan_tracked_channels = AsyncMock(return_value={"scanned_channels": 3, "new_reels_processed": 2})
    context.application.bot_data = {"scheduler": scheduler}

    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    status_msg = MagicMock(spec=Message)
    status_msg.edit_text = AsyncMock()
    message.reply_text = AsyncMock(return_value=status_msg)
    update.message = message

    await handle_check_now(update, context)
    scheduler.scan_tracked_channels.assert_awaited_once()
    status_msg.edit_text.assert_awaited_once()
    assert "Scanned <b>3</b> creators" in status_msg.edit_text.call_args[0][0]
    assert "indexed <b>2</b> new Reels" in status_msg.edit_text.call_args[0][0]


@pytest.mark.asyncio
async def test_track_with_profile_url_and_untrack(temp_db: ReelDatabase):
    context = MagicMock()
    pipeline = MagicMock()
    pipeline.db = temp_db
    context.application.bot_data = {"pipeline": pipeline}

    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    update.message = message

    # Track with full Instagram URL
    context.args = ["https://www.instagram.com/hubermanlab/?hl=en"]
    await handle_track(update, context)
    assert temp_db.get_tracked_channel("hubermanlab") is not None
    assert temp_db.get_tracked_channel("@hubermanlab") is not None
    assert "hubermanlab" in message.reply_text.call_args[0][0]

    # Untrack with plain handle
    message.reply_text.reset_mock()
    context.args = ["hubermanlab"]
    await handle_untrack(update, context)
    assert temp_db.get_tracked_channels(active_only=True) == []
    assert "Stopped monitoring" in message.reply_text.call_args[0][0]


def test_downloader_get_channel_reels_deduplication():
    downloader = Downloader()
    mock_info = {
        "entries": [
            {"id": "C-dup1", "url": "https://www.instagram.com/reel/C-dup1/"},
            {"id": "C-dup1", "url": "https://www.instagram.com/reel/C-dup1/"},  # duplicate
            {"id": "C-dup2", "url": "https://www.instagram.com/p/C-dup2/"},    # post format
        ]
    }
    with patch("modules.downloader.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        reels = downloader.get_channel_reels("https://instagram.com/test_creator/", limit=5)
        assert len(reels) == 2
        assert reels[0] == "https://www.instagram.com/reel/C-dup1/"
        assert reels[1] == "https://www.instagram.com/reel/C-dup2/"


@pytest.mark.asyncio
async def test_check_now_handles_exception():
    context = MagicMock()
    scheduler = MagicMock()
    scheduler.scan_tracked_channels = AsyncMock(side_effect=RuntimeError("Scrape blocked"))
    context.application.bot_data = {"scheduler": scheduler}

    update = MagicMock(spec=Update)
    message = MagicMock(spec=Message)
    status_msg = MagicMock(spec=Message)
    status_msg.edit_text = AsyncMock()
    message.reply_text = AsyncMock(return_value=status_msg)
    update.message = message

    await handle_check_now(update, context)
    status_msg.edit_text.assert_awaited_once()
    assert "Scrape blocked" in status_msg.edit_text.call_args[0][0]
