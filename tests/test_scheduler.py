import pytest
from unittest.mock import AsyncMock, MagicMock
from modules.scheduler import SchedulerService


@pytest.mark.asyncio
async def test_digest_and_canary():
    mock_db = MagicMock()
    mock_db.get_recent_reels.return_value = [
        {"title": "High-Protein Oats", "category": "recipe", "url": "https://instagram.com/reel/1"},
        {"title": "Deadlift Form", "category": "workout", "url": "https://instagram.com/reel/2"},
        {"title": "Docker Setup", "category": "tech", "url": "https://instagram.com/reel/3"},
        {"title": "Book Summary", "category": "idea", "url": "https://instagram.com/reel/4"}
    ]
    mock_pipeline = MagicMock()
    mock_pipeline.process_url = AsyncMock(return_value=(True, "Success", 1))
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()

    service = SchedulerService(
        db=mock_db,
        bot=mock_bot,
        chat_id="-100123456",
        pipeline=mock_pipeline,
        canary_url="https://instagram.com/reel/test/"
    )
    digest = service.build_weekly_digest()
    assert digest is not None
    assert "🧠 <b>YOUR REELMIND — SUNDAY REVIEW</b>" in digest
    assert "High-Protein Oats" in digest
    assert "Deadlift Form" in digest
    assert "Docker Setup" in digest

    # Test canary pass
    ok, msg = await service.run_canary_test()
    assert ok is True
    assert "passed" in msg.lower()

    # Test canary failure sends alert
    mock_pipeline.process_url = AsyncMock(return_value=(False, "Download 404", None))
    ok_fail, msg_fail = await service.run_canary_test()
    assert ok_fail is False
    assert "Download 404" in msg_fail
    mock_bot.send_message.assert_awaited()


def test_empty_digest():
    mock_db = MagicMock()
    mock_db.get_recent_reels.return_value = []
    service = SchedulerService(
        db=mock_db,
        bot=MagicMock(),
        chat_id="-100"
    )
    assert service.build_weekly_digest() is None
