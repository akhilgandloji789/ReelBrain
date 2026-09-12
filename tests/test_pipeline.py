import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from pipeline import ReelPipeline
from modules.analyzer import ReelAnalysisOutput, EntityItem


@pytest.mark.asyncio
async def test_pipeline_state_machine_flow(tmp_path: Path):
    settings = MagicMock()
    settings.TEMP_DIR = tmp_path
    settings.allowed_users = []
    settings.EMBEDDING_MODEL = "models/gemini-embedding-001"
    settings.EMBEDDING_DIM = 768
    
    mock_db = MagicMock()
    mock_db.is_processed.return_value = False
    mock_db.add_reel.return_value = 1
    
    mock_downloader = MagicMock()
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"dummy")
    mock_downloader.parse_input.return_value = ("C-test1", "https://instagram.com/reel/C-test1/", "Meal prep intent")
    mock_downloader.download_video.return_value = video_file
    
    mock_analyzer = MagicMock()
    output = ReelAnalysisOutput(
        title="Test Reel",
        category="tech",
        tldr="Summary",
        entities=[EntityItem(entity_type="fact", text="Fact 1", start_ts=2.0)],
        keyframe_timestamps=[1.0]
    )
    mock_analyzer.analyze_video.return_value = output
    mock_analyzer.generate_embedding.return_value = [0.1, 0.2]
    
    mock_extractor = MagicMock()
    fake_frame = tmp_path / "frame.jpg"
    fake_frame.write_bytes(b"frame")
    mock_extractor.extract_frames.return_value = [fake_frame]
    
    mock_publisher = AsyncMock()
    mock_publisher.publish_reel.return_value = 10
    
    pipeline = ReelPipeline(
        settings=settings,
        db=mock_db,
        downloader=mock_downloader,
        analyzer=mock_analyzer,
        extractor=mock_extractor,
        publisher=mock_publisher
    )
    
    success, msg, thread_id = await pipeline.process_url("https://instagram.com/reel/C-test1/ Meal prep intent")
    assert success is True
    assert "Test Reel" in msg
    assert thread_id == 10
    
    mock_db.add_reel.assert_called_once()
    assert mock_db.update_reel_status.call_count >= 4
    # State transitions: DOWNLOADING, ANALYZING, EXTRACTING_FRAMES, PUBLISHING, INDEXING, COMPLETED
    assert not video_file.exists()
    mock_extractor.cleanup_files.assert_called_once_with([fake_frame])


@pytest.mark.asyncio
async def test_pipeline_unauthorized_user():
    settings = MagicMock()
    settings.allowed_users = [111, 222]
    
    pipeline = ReelPipeline(settings=settings, db=MagicMock(), downloader=MagicMock())
    success, msg, thread_id = await pipeline.process_url("https://instagram.com/reel/C-test/", user_id=999)
    assert success is False
    assert "Unauthorized" in msg


@pytest.mark.asyncio
async def test_pipeline_idempotency():
    settings = MagicMock()
    settings.allowed_users = []
    
    mock_db = MagicMock()
    mock_db.is_processed.return_value = True
    mock_db.get_reel_by_shortcode.return_value = {"title": "Existing Reel", "category": "recipe"}
    
    mock_downloader = MagicMock()
    mock_downloader.parse_input.return_value = ("C-exist", "https://instagram.com/reel/C-exist/", None)
    
    pipeline = ReelPipeline(settings=settings, db=mock_db, downloader=mock_downloader)
    success, msg, thread_id = await pipeline.process_url("https://instagram.com/reel/C-exist/")
    assert success is True
    assert "Already saved" in msg
    assert "Existing Reel" in msg


def test_startup_recovery(tmp_path: Path):
    settings = MagicMock()
    settings.TEMP_DIR = tmp_path
    
    old_file = tmp_path / "old.mp4"
    old_file.write_bytes(b"old")
    # Make file look 2 hours old
    two_hours_ago = time.time() - 7200
    import os
    os.utime(str(old_file), (two_hours_ago, two_hours_ago))

    fresh_file = tmp_path / "fresh.mp4"
    fresh_file.write_bytes(b"fresh")

    pipeline = ReelPipeline(settings=settings, db=MagicMock(), downloader=MagicMock())
    pipeline.startup_recovery()

    assert not old_file.exists()
    assert fresh_file.exists()
