import re
import time
from pathlib import Path
from typing import Any
from yt_dlp import YoutubeDL
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Matches instagram reel or post URLs with optional trailing slashes and query parameters
REEL_REGEX = re.compile(r'https?://(?:www\.)?(?:instagram\.com|instagr\.am)/(?:reel|reels|p)/([a-zA-Z0-9_-]+)(?:/[^\s]*|\?[^\s]*)?')


class DownloadError(Exception):
    pass


class CircuitBreakerOpen(Exception):
    pass


class Downloader:
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.consecutive_failures = 0
        self.circuit_open_until = 0.0

    def parse_input(self, raw_text: str) -> tuple[str | None, str | None, str | None]:
        match = REEL_REGEX.search(raw_text)
        if not match:
            return None, None, None
        
        shortcode = match.group(1)
        canonical = f"https://www.instagram.com/reel/{shortcode}/"
        
        # Everything outside the full matched URL is treated as user intent
        intent = raw_text.replace(match.group(0), "").strip()
        intent = intent if intent else None
        return shortcode, canonical, intent

    def _check_circuit(self) -> None:
        now = time.time()
        if now < self.circuit_open_until:
            remaining = int(self.circuit_open_until - now)
            raise CircuitBreakerOpen(f"Downloader circuit breaker active for {remaining}s due to consecutive failures.")

    def _record_success(self) -> None:
        self.consecutive_failures = 0

    def _record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.circuit_open_until = time.time() + self.reset_timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(DownloadError),
        reraise=True
    )
    def download_video(self, url: str, output_dir: Path | str) -> Path:
        self._check_circuit()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        outtmpl = str(out_path / "%(id)s.%(ext)s")
        ydl_opts = {
            "outtmpl": outtmpl,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info.get("id") if info else None
                if not video_id:
                    # Fallback to shortcode
                    match = REEL_REGEX.search(url)
                    video_id = match.group(1) if match else "reel"

                expected_mp4 = out_path / f"{video_id}.mp4"
                if expected_mp4.exists():
                    self._record_success()
                    return expected_mp4
                
                # Check for any downloaded file with video_id prefix
                for f in out_path.glob(f"{video_id}.*"):
                    self._record_success()
                    return f
                    
                raise DownloadError(f"Video file not found after download for {url}")
        except Exception as e:
            self._record_failure()
            # Clean up any partial files created during failed download
            for pattern in ("*.part*", "*.ytdl", "*.temp*", "*.tmp*"):
                for partial in out_path.glob(pattern):
                    try:
                        partial.unlink()
                    except Exception:
                        pass
            raise DownloadError(f"Download failure on {url}: {str(e)}") from e

    def get_channel_reels(self, handle: str, limit: int = 5) -> list[str]:
        clean_handle = handle.strip().lstrip("@")
        if "/" in clean_handle:
            clean_handle = clean_handle.split("?")[0].rstrip("/")
            parts = [p for p in clean_handle.split("/") if p and "instagram.com" not in p and "instagr.am" not in p and "http" not in p]
            if parts:
                clean_handle = parts[-1]
        if not clean_handle:
            return []

        ydl_opts = {
            "extract_flat": True,
            "quiet": True,
            "no_warnings": True,
            "playlist_items": f"1-{limit}",
        }
        target_urls = [
            f"https://www.instagram.com/{clean_handle}/reels/",
            f"https://www.instagram.com/{clean_handle}/",
        ]

        reels: list[str] = []
        for target_url in target_urls:
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(target_url, download=False)
                    entries = info.get("entries") or [] if info else []
                    for entry in entries:
                        if not entry:
                            continue
                        url = entry.get("url")
                        entry_id = entry.get("id")
                        target_reel = None
                        if url:
                            match = REEL_REGEX.search(url)
                            if match:
                                target_reel = f"https://www.instagram.com/reel/{match.group(1)}/"
                            elif entry_id:
                                target_reel = f"https://www.instagram.com/reel/{entry_id}/"
                            else:
                                target_reel = url
                        elif entry_id:
                            target_reel = f"https://www.instagram.com/reel/{entry_id}/"

                        if target_reel and target_reel not in reels:
                            reels.append(target_reel)
                            if len(reels) >= limit:
                                break
                    if reels:
                        break
            except Exception:
                continue
        return reels
