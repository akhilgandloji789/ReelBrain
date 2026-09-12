import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.downloader import Downloader, DownloadError, CircuitBreakerOpen


def test_parse_input_with_intent():
    downloader = Downloader()
    text = "https://www.instagram.com/reel/C-12345Abc/ Try this for Sunday meal prep"
    sid, canonical, intent = downloader.parse_input(text)
    assert sid == "C-12345Abc"
    assert canonical == "https://www.instagram.com/reel/C-12345Abc/"
    assert intent == "Try this for Sunday meal prep"


def test_parse_input_variations():
    downloader = Downloader()
    
    # Query params and no trailing intent
    sid, canonical, intent = downloader.parse_input("Check this out: https://instagram.com/reels/C-xyz999?igsh=12345")
    assert sid == "C-xyz999"
    assert canonical == "https://www.instagram.com/reel/C-xyz999/"
    assert intent == "Check this out:"

    # Post URL /p/
    sid, canonical, intent = downloader.parse_input("https://www.instagram.com/p/C-post123/")
    assert sid == "C-post123"
    assert canonical == "https://www.instagram.com/reel/C-post123/"
    assert intent is None

    # Invalid URL
    sid, canonical, intent = downloader.parse_input("https://youtube.com/watch?v=123")
    assert sid is None
    assert canonical is None
    assert intent is None


def test_download_video_success(tmp_path: Path):
    downloader = Downloader()
    fake_file = tmp_path / "fake_id.mp4"
    fake_file.write_bytes(b"dummy mp4 data")

    with patch("modules.downloader.YoutubeDL") as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"id": "fake_id"}

        result_path = downloader.download_video("https://www.instagram.com/reel/fake_id/", tmp_path)
        assert result_path.exists()
        assert result_path.name == "fake_id.mp4"
        assert downloader.consecutive_failures == 0


def test_circuit_breaker_trips(tmp_path: Path):
    downloader = Downloader()
    
    with patch("modules.downloader.YoutubeDL") as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = Exception("Network error")

        # Tenacity will retry 3 times per download_video call
        with pytest.raises(DownloadError):
            downloader.download_video("https://www.instagram.com/reel/fail1/", tmp_path)
        assert downloader.consecutive_failures >= 1

        # Trip circuit manually to verify exception
        downloader.consecutive_failures = 3
        downloader.circuit_open_until = 9999999999.0
        
        with pytest.raises(CircuitBreakerOpen):
            downloader.download_video("https://www.instagram.com/reel/fail2/", tmp_path)
