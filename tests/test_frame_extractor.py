import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from modules.frame_extractor import FrameExtractor


@patch("subprocess.run")
def test_extract_frames_ffmpeg(mock_run, tmp_path: Path):
    mock_run.return_value = MagicMock(returncode=0)
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy")
    
    extractor = FrameExtractor()
    frames_dir = tmp_path / "frames"
    
    # Simulate ffmpeg outputting the files
    def fake_run(cmd, *args, **kwargs):
        out_file = Path(cmd[-1])
        out_file.write_bytes(b"fake image data")
        return MagicMock(returncode=0)
        
    mock_run.side_effect = fake_run

    frames = extractor.extract_frames(video, [5.0, 12.0], frames_dir)
    assert len(frames) == 2
    assert all(f.exists() for f in frames)

    # Test cleanup
    extractor.cleanup_files(frames)
    assert not any(f.exists() for f in frames)


def test_extract_frames_opencv_fallback(tmp_path: Path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy")
    
    extractor = FrameExtractor()
    frames_dir = tmp_path / "frames"

    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")), \
         patch("modules.frame_extractor.cv2") as mock_cv2:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, "fake_frame_matrix")
        mock_cv2.VideoCapture.return_value = mock_cap
        
        def fake_imwrite(path, frame):
            Path(path).write_bytes(b"opencv frame data")
            return True
            
        mock_cv2.imwrite.side_effect = fake_imwrite

        frames = extractor.extract_frames(video, [3.0, 7.5], frames_dir)
        assert len(frames) == 2
        assert all(f.exists() for f in frames)
