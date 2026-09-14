import asyncio
import time
from pathlib import Path
from typing import Any

from config import Settings
from modules.storage import ReelDatabase
from modules.downloader import Downloader
from modules.analyzer import ReelAnalyzer
from modules.frame_extractor import FrameExtractor
from modules.publisher import TelegramPublisher
from modules.metrics import TelemetryMetrics


class ReelPipeline:
    def __init__(
        self,
        settings: Settings,
        db: ReelDatabase | None = None,
        downloader: Downloader | None = None,
        analyzer: ReelAnalyzer | None = None,
        extractor: FrameExtractor | None = None,
        publisher: TelegramPublisher | None = None,
        metrics: TelemetryMetrics | None = None,
    ):
        self.settings = settings
        self.db = db or ReelDatabase(settings.DATA_DIR / "reelminds.db")
        self.downloader = downloader or Downloader()
        self.analyzer = analyzer or ReelAnalyzer(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dim=settings.EMBEDDING_DIM
        )
        self.extractor = extractor or FrameExtractor()
        self.publisher = publisher or TelegramPublisher(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            group_chat_id=settings.TELEGRAM_GROUP_CHAT_ID,
            topic_map=settings.topic_map
        )
        self.metrics = metrics or TelemetryMetrics()
        self._lock = asyncio.Lock()

    def startup_recovery(self) -> None:
        """Purge orphaned temporary files and reset stale jobs on startup."""
        temp_dir = self.settings.TEMP_DIR
        now = time.time()
        if temp_dir.exists():
            for p in temp_dir.rglob("*"):
                if p.is_file() and (now - p.stat().st_mtime > 3600):  # older than 1 hr
                    try:
                        p.unlink()
                    except Exception:
                        pass
        try:
            self.db.cleanup_orphaned_jobs(max_age_hours=1)
        except Exception:
            pass

    async def process_url(self, raw_input: str, user_id: int | None = None) -> tuple[bool, str, int | None]:
        start_time = time.time()
        if user_id is not None and self.settings.allowed_users and user_id not in self.settings.allowed_users:
            return False, "⛔ Unauthorized user. Access restricted by administrator.", None

        async with self._lock:
            shortcode, canonical_url, user_intent = self.downloader.parse_input(raw_input)
            if not shortcode or not canonical_url:
                return False, "Not a valid Instagram Reel or Post URL.", None

            # Idempotency check
            if self.db.is_processed(shortcode):
                existing = self.db.get_reel_by_shortcode(shortcode)
                cat = existing.get("category", "General") if existing else "General"
                title = existing.get("title", "Reel") if existing else "Reel"
                return True, f"⚠️ Already saved under <b>#{cat}</b>:\n📌 <i>{title}</i>", None

            video_dir = self.settings.TEMP_DIR / "videos"
            frames_dir = self.settings.TEMP_DIR / "frames"
            video_file: Path | None = None
            frame_paths: list[Path] = []
            reel_id: int | None = None

            try:
                # State: RECEIVED -> register in DB
                reel_id = self.db.add_reel(
                    url=canonical_url,
                    shortcode=shortcode,
                    category="other",
                    title="Processing...",
                    user_intent=user_intent
                )

                # State: DOWNLOADING
                self.db.update_reel_status(reel_id, "DOWNLOADING")
                video_file = self.downloader.download_video(canonical_url, output_dir=video_dir)

                # State: DOWNLOADED -> ANALYZING
                self.db.update_reel_status(reel_id, "ANALYZING")
                analysis = self.analyzer.analyze_video(video_file)

                # State: ANALYZED -> EXTRACTING_FRAMES
                self.db.update_reel_status(reel_id, "EXTRACTING_FRAMES")
                frame_paths = self.extractor.extract_frames(
                    video_path=video_file,
                    timestamps=analysis.keyframe_timestamps,
                    output_dir=frames_dir
                )

                # Generate Embedding
                embedding: list[float] | None = None
                try:
                    embed_text = f"{analysis.title}\n{user_intent or ''}\n{analysis.tldr}\n" + " ".join([e.text for e in analysis.entities])
                    embedding = self.analyzer.generate_embedding(embed_text)
                except Exception:
                    pass

                # State: PUBLISHING
                self.db.update_reel_status(reel_id, "PUBLISHING")
                thread_id = await self.publisher.publish_reel(
                    reel_id=reel_id,
                    analysis=analysis,
                    frame_paths=frame_paths,
                    original_url=canonical_url,
                    user_intent=user_intent
                )

                # State: INDEXING
                self.db.update_reel_status(reel_id, "INDEXING")
                self.db.update_reel_metadata(
                    reel_id=reel_id,
                    category=analysis.category,
                    title=analysis.title,
                    raw_transcript=analysis.tldr
                )
                self.db.add_entities(reel_id, [e.model_dump() for e in analysis.entities])

                if embedding:
                    self.db.store_embedding(
                        reel_id=reel_id,
                        model=self.settings.EMBEDDING_MODEL,
                        dim=self.settings.EMBEDDING_DIM,
                        vector=embedding
                    )

                # State: COMPLETED
                self.db.update_reel_status(reel_id, "COMPLETED")

                elapsed = time.time() - start_time
                self.metrics.record_run(elapsed, success=True)
                return True, f"✅ Indexed <b>{analysis.title}</b> under <i>#{analysis.category}</i> ({round(elapsed, 1)}s)", thread_id

            except Exception as e:
                if reel_id:
                    self.db.update_reel_status(reel_id, "FAILED", error_message=str(e))
                elapsed = time.time() - start_time
                self.metrics.record_run(elapsed, success=False)
                return False, f"⚠️ Failed to process reel: {str(e)}", None

            finally:
                if getattr(self.settings, "CLEANUP_TEMP", True):
                    if video_file and video_file.exists():
                        try:
                            video_file.unlink()
                        except Exception:
                            pass
                    if frame_paths:
                        self.extractor.cleanup_files(frame_paths)
                    if shortcode and video_dir.exists():
                        for aux in video_dir.glob(f"*{shortcode}*"):
                            try:
                                aux.unlink()
                            except Exception:
                                pass
