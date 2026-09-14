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
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 120, session_id: str | None = None):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.consecutive_failures = 0
        self.circuit_open_until = 0.0
        self.session_id = session_id

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
        if self.session_id:
            ydl_opts["http_headers"] = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                "Cookie": f"sessionid={self.session_id};",
                "Accept-Language": "en-US,en;q=0.9",
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
            # Fallback: Attempt download via instagrapi if session is present
            if self.session_id:
                try:
                    from instagrapi import Client as InstaClient
                    cl = InstaClient()
                    cl.login_by_sessionid(self.session_id)
                    downloaded_file = cl.video_download_by_url(url, folder=out_path)
                    if downloaded_file and Path(downloaded_file).exists():
                        self._record_success()
                        return Path(downloaded_file)
                except Exception:
                    pass

            self._record_failure()
            # Clean up any partial files created during failed download
            for pattern in ("*.part*", "*.ytdl", "*.temp*", "*.tmp*"):
                for partial in out_path.glob(pattern):
                    try:
                        partial.unlink()
                    except Exception:
                        pass
            err_str = str(e)
            if "empty media response" in err_str:
                err_str = "Instagram requires login to access this media. Check or refresh INSTAGRAM_SESSION_ID."
            raise DownloadError(f"Download failure on {url}: {err_str}") from e

    def get_channel_reels(self, handle: str, limit: int = 5) -> list[str]:
        clean_handle = handle.strip().lstrip("@")
        if "/" in clean_handle:
            clean_handle = clean_handle.split("?")[0].rstrip("/")
            parts = [p for p in clean_handle.split("/") if p and "instagram.com" not in p and "instagr.am" not in p and "http" not in p]
            if parts:
                clean_handle = parts[-1]
        if not clean_handle:
            return []

        # 1. Primary: If instagrapi session is available, query mobile API for 100% reliability
        if self.session_id:
            try:
                from instagrapi import Client as InstaClient
                cl = InstaClient()
                cl.login_by_sessionid(self.session_id)
                user_id = cl.user_id_from_username(clean_handle)
                if user_id:
                    clips = cl.user_clips(user_id, amount=limit)
                    reels: list[str] = []
                    for clip in clips:
                        code = getattr(clip, "code", None)
                        if code:
                            reels.append(f"https://www.instagram.com/reel/{code}/")
                    if reels:
                        return reels
            except Exception:
                pass

        # 2. Fallback: yt-dlp flat playlist extraction
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
