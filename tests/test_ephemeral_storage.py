import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from config import Settings
from modules.storage import ReelDatabase
from modules.analyzer import ReelAnalysisOutput, EntityItem
from pipeline import ReelPipeline
from modules.downloader import Downloader, DownloadError
from main import get_temp_disk_usage_mb


@pytest.fixture
def ephemeral_env(tmp_path: Path):
    data_dir = tmp_path / "data"
    temp_dir = tmp_path / "temp"
    data_dir.mkdir()
    temp_dir.mkdir()

    settings = Settings(
        TELEGRAM_BOT_TOKEN="mock:token",
        TELEGRAM_GROUP_CHAT_ID="-1001",
        GEMINI_API_KEY="mock_key",
        DATA_DIR=data_dir,
        TEMP_DIR=temp_dir,
        CLEANUP_TEMP=True
    )
    db = ReelDatabase(data_dir / "test_ephemeral.db")
    return settings, db, temp_dir


def test_temp_disk_usage_mb_calculation(tmp_path: Path):
    temp_dir = tmp_path / "temp_calc"
    temp_dir.mkdir()
    assert get_temp_disk_usage_mb(temp_dir) == 0.0

    # Create 2 MB file
    dummy = temp_dir / "test.bin"
    dummy.write_bytes(b"\0" * (2 * 1024 * 1024))
    assert get_temp_disk_usage_mb(temp_dir) == 2.0

    dummy.unlink()
    assert get_temp_disk_usage_mb(temp_dir) == 0.0


@pytest.mark.asyncio
async def test_ephemeral_cleanup_on_pipeline_success(ephemeral_env):
    settings, db, temp_dir = ephemeral_env

    # Mock downloader to create a dummy video file
    downloader = MagicMock()
    downloader.parse_input.return_value = ("C-eph123", "https://www.instagram.com/reel/C-eph123/", "Intent")
    
    video_dir = temp_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    fake_video = video_dir / "fake_vid.mp4"

    def fake_download(*args, **kwargs):
        fake_video.write_text("dummy video data")
        return fake_video

    downloader.download_video.side_effect = fake_download

    # Mock analyzer
    analyzer = MagicMock()
    analyzer.analyze_video.return_value = ReelAnalysisOutput(
        title="Ephemeral Test",
        category="tech",
        tldr="Testing zero disk footprint",
        entities=[EntityItem(entity_type="fact", text="Zero local disk", start_ts=1.0)],
        keyframe_timestamps=[1.0, 2.0]
    )
    analyzer.generate_embedding.return_value = [0.1] * 768

    # Mock extractor to create dummy frame files
    extractor = MagicMock()
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    fake_frame1 = frames_dir / "f1.jpg"
    fake_frame2 = frames_dir / "f2.jpg"

    def fake_extract(*args, **kwargs):
        fake_frame1.write_text("frame1")
        fake_frame2.write_text("frame2")
        return [fake_frame1, fake_frame2]

    extractor.extract_frames.side_effect = fake_extract
    extractor.cleanup_files.side_effect = lambda paths: [Path(p).unlink(missing_ok=True) for p in paths]

    # Mock publisher
    publisher = MagicMock()
    publisher.publish_reel = AsyncMock(return_value=123)

    pipeline = ReelPipeline(
        settings=settings,
        db=db,
        downloader=downloader,
        analyzer=analyzer,
        extractor=extractor,
        publisher=publisher
    )

    success, msg, thread_id = await pipeline.process_url("https://www.instagram.com/reel/C-eph123/")
    assert success is True

    # Assert video and frame files are automatically deleted from local disk (Ephemeral mode)
    assert not fake_video.exists()
    assert not fake_frame1.exists()
    assert not fake_frame2.exists()
    assert get_temp_disk_usage_mb(temp_dir) == 0.0


@pytest.mark.asyncio
async def test_ephemeral_cleanup_on_pipeline_failure(ephemeral_env):
    settings, db, temp_dir = ephemeral_env

    downloader = MagicMock()
    downloader.parse_input.return_value = ("C-fail123", "https://www.instagram.com/reel/C-fail123/", None)
    
    video_dir = temp_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    fake_video = video_dir / "fail_vid.mp4"

    def fake_download(*args, **kwargs):
        fake_video.write_text("dummy video data before fail")
        return fake_video

    downloader.download_video.side_effect = fake_download

    # Analyzer fails
    analyzer = MagicMock()
    analyzer.analyze_video.side_effect = RuntimeError("Gemini API Rate Limit")

    extractor = MagicMock()
    extractor.cleanup_files.side_effect = lambda paths: [Path(p).unlink(missing_ok=True) for p in paths]

    pipeline = ReelPipeline(
        settings=settings,
        db=db,
        downloader=downloader,
        analyzer=analyzer,
        extractor=extractor,
        publisher=MagicMock()
    )

    success, msg, thread_id = await pipeline.process_url("https://www.instagram.com/reel/C-fail123/")
    assert success is False
    assert "Gemini API Rate Limit" in msg

    # Assert video file is still cleaned up in finally block
    assert not fake_video.exists()
    assert get_temp_disk_usage_mb(temp_dir) == 0.0


def test_downloader_cleans_partial_files_on_failure(tmp_path: Path):
    downloader = Downloader()
    out_dir = tmp_path / "videos_failed"
    out_dir.mkdir()

    with patch("modules.downloader.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        def fail_extract(*args, **kwargs):
            # simulate partial file left behind by yt-dlp before exception
            (out_dir / "test.mp4.part").write_text("partial data")
            (out_dir / "test.ytdl").write_text("ytdl temp data")
            raise RuntimeError("Network reset by peer")

        mock_instance.extract_info.side_effect = fail_extract
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        with pytest.raises(DownloadError):
            downloader.download_video("https://www.instagram.com/reel/C-dummy/", output_dir=out_dir)

        # Confirm partial and temp files are purged
        assert not (out_dir / "test.mp4.part").exists()
        assert not (out_dir / "test.ytdl").exists()


def test_downloader_cleans_temporary_files_on_failure(tmp_path: Path):
    downloader = Downloader()
    out_dir = tmp_path / "videos_failed_tmp"
    out_dir.mkdir()

    with patch("modules.downloader.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        def fail_extract(*args, **kwargs):
            (out_dir / "stream.temp").write_text("temp data")
            (out_dir / "video.tmp").write_text("tmp data")
            raise RuntimeError("Extraction failed")

        mock_instance.extract_info.side_effect = fail_extract
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        with pytest.raises(DownloadError):
            downloader.download_video("https://www.instagram.com/reel/C-dummy2/", output_dir=out_dir)

        assert not (out_dir / "stream.temp").exists()
        assert not (out_dir / "video.tmp").exists()


@pytest.mark.asyncio
async def test_pipeline_sweeps_auxiliary_files_with_shortcode(ephemeral_env):
    settings, db, temp_dir = ephemeral_env
    video_dir = temp_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create auxiliary files left behind during processing
    aux1 = video_dir / "C-auxShortcode.f137.mp4"
    aux1.write_text("aux video")
    aux2 = video_dir / "C-auxShortcode.f140.m4a"
    aux2.write_text("aux audio")

    downloader = MagicMock()
    downloader.parse_input.return_value = ("C-auxShortcode", "https://www.instagram.com/reel/C-auxShortcode/", None)
    downloader.download_video.side_effect = RuntimeError("Fail mid-process")

    pipeline = ReelPipeline(settings=settings, db=db, downloader=downloader)
    success, _, _ = await pipeline.process_url("https://www.instagram.com/reel/C-auxShortcode/")
    assert success is False

    # Verify shortcode auxiliary files were swept in finally block
    assert not aux1.exists()
    assert not aux2.exists()
    assert get_temp_disk_usage_mb(temp_dir) == 0.0
