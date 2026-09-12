import subprocess
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


class FrameExtractor:
    def extract_frames(self, video_path: Path | str, timestamps: list[float], output_dir: Path | str) -> list[Path]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_paths: list[Path] = []
        video_stem = video_path.stem

        for idx, ts in enumerate(timestamps[:4]):
            output_frame = output_dir / f"{video_stem}_frame_{idx}_{int(ts)}s.jpg"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(output_frame)
            ]
            success = False
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                if output_frame.exists():
                    extracted_paths.append(output_frame)
                    success = True
            except Exception:
                pass

            if not success and cv2 is not None:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            cv2.imwrite(str(output_frame), frame)
                            if output_frame.exists():
                                extracted_paths.append(output_frame)
                        cap.release()
                except Exception:
                    pass

        return extracted_paths

    def cleanup_files(self, paths: list[Path | str]) -> None:
        for p in paths:
            try:
                path_obj = Path(p)
                if path_obj.exists():
                    path_obj.unlink()
            except Exception:
                pass
