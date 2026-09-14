import logging
import re
from typing import Any
import httpx
from config import Settings
from modules.downloader import Downloader, REEL_REGEX
from modules.storage import ReelDatabase

logger = logging.getLogger("instagram_receiver")


class InstagramReceiver:
    def __init__(
        self,
        pipeline: Any,
        db: ReelDatabase,
        session_id: str | None = None,
        verify_token: str | None = None,
        downloader: Downloader | None = None,
        http_client: Any = None
    ):
        self.pipeline = pipeline
        self.db = db
        self.session_id = session_id
        self.verify_token = verify_token
        if downloader:
            self.downloader = downloader
        elif getattr(pipeline, "downloader", None) and isinstance(pipeline.downloader, Downloader):
            self.downloader = pipeline.downloader
        else:
            self.downloader = Downloader()
        self._http_client = http_client

    def verify_webhook(
        self,
        mode: str | None,
        token: str | None,
        challenge: str | None
    ) -> str | None:
        """Verify Meta Graph webhook subscription handshake."""
        if mode == "subscribe" and token and self.verify_token and token == self.verify_token:
            logger.info("Meta Instagram Webhook verified successfully.")
            return challenge
        logger.warning(f"Webhook verification rejected (mode={mode}, token_match={token == self.verify_token})")
        return None

    def extract_messages_from_webhook(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse Meta Messenger/Instagram webhook JSON and extract DM reel items."""
        messages: list[dict[str, Any]] = []
        if not isinstance(payload, dict):
            return messages

        entries = payload.get("entry", [])
        for entry in entries:
            messaging = entry.get("messaging", [])
            for event in messaging:
                sender_id = event.get("sender", {}).get("id")
                msg_obj = event.get("message", {})
                mid = msg_obj.get("mid")
                if not mid:
                    continue

                text = msg_obj.get("text", "") or ""
                target_url = None
                intent = None

                # Check text for reel URL
                match = REEL_REGEX.search(text)
                if match:
                    shortcode = match.group(1)
                    target_url = f"https://www.instagram.com/reel/{shortcode}/"
                    intent = text.replace(match.group(0), "").strip() or None

                # Check attachments (shares)
                attachments = msg_obj.get("attachments", [])
                for att in attachments:
                    att_payload = att.get("payload", {})
                    url_val = att_payload.get("url", "")
                    if url_val:
                        att_match = REEL_REGEX.search(url_val)
                        if att_match:
                            target_url = f"https://www.instagram.com/reel/{att_match.group(1)}/"
                            break

                # If target_url came from attachment and text was present, use text as user intent
                if not intent and text and target_url:
                    intent = text.strip() or None

                if target_url:
                    messages.append({
                        "mid": mid,
                        "sender_id": sender_id,
                        "url": target_url,
                        "intent": intent,
                        "raw_text": text
                    })

        return messages

    async def process_message(
        self,
        mid: str,
        text: str = "",
        sender_id: str | None = None,
        url: str | None = None
    ) -> tuple[bool, str, int | None]:
        """Deduplicate by mid, extract reel URL, and pass to ReelPipeline."""
        if not mid:
            import hashlib
            mid = hashlib.sha256(f"{sender_id}:{text}:{url}".encode("utf-8")).hexdigest()[:24]

        if self.db.is_dm_processed(mid):
            return False, f"Message {mid} already processed.", None

        reel_url = url
        intent = ""

        if not reel_url and text:
            shortcode, canonical, parsed_intent = self.downloader.parse_input(text)
            if canonical:
                reel_url = canonical
                intent = parsed_intent or ""
        elif text and not intent:
            match = REEL_REGEX.search(text)
            if match:
                intent = text.replace(match.group(0), "").strip()
            else:
                intent = text.strip()

        if not reel_url:
            return False, "No Instagram Reel found in message.", None

        dm_intent = f"Instagram DM from {sender_id or 'user'}"
        if intent:
            dm_intent = f"{dm_intent}: {intent}"

        raw_input = f"{reel_url} {dm_intent}"
        success, msg, thread_id = await self.pipeline.process_url(raw_input)

        self.db.record_processed_dm(mid=mid, sender_id=sender_id, url=reel_url)
        return success, msg, thread_id

    async def handle_webhook(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Process all reel shares in a Meta webhook payload."""
        extracted = self.extract_messages_from_webhook(payload)
        results = []
        for item in extracted:
            success, msg, thread_id = await self.process_message(
                mid=item["mid"],
                text=item["raw_text"],
                sender_id=item["sender_id"],
                url=item["url"]
            )
            results.append({
                "mid": item["mid"],
                "success": success,
                "message": msg,
                "thread_id": thread_id
            })
        return results

    async def poll_direct_inbox(self) -> list[tuple[bool, str, int | None]]:
        """Poll Instagram Direct v2 inbox via private session cookies."""
        if not self.session_id:
            return []

        url = "https://i.instagram.com/api/v1/direct_v2/inbox/"
        headers = {
            "User-Agent": "Instagram 219.0.0.12.117 Android (30/11; 480dpi; 1080x2240; Xiaomi; Redmi Note 7; lavender; qcom; en_US)",
            "Cookie": f"sessionid={self.session_id}",
            "Accept-Language": "en-US",
            "Accept-Encoding": "gzip, deflate",
        }

        results: list[tuple[bool, str, int | None]] = []

        try:
            client = self._http_client or httpx.AsyncClient(timeout=15.0)
            close_client = self._http_client is None
            try:
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    logger.warning(f"Failed to poll Instagram inbox: HTTP {response.status_code}")
                    return results

                data = response.json()
                threads = data.get("inbox", {}).get("threads", [])
                for thread in threads:
                    items = thread.get("items", [])
                    for item in items:
                        mid = str(item.get("item_id", ""))
                        if not mid or self.db.is_dm_processed(mid):
                            continue

                        sender_id = str(item.get("user_id", ""))
                        item_type = item.get("item_type", "")
                        reel_url = None
                        text_val = item.get("text", "") or ""

                        if item_type in ("media_share", "clip", "reel_share", "felix_share"):
                            media = item.get("media")
                            if not media:
                                clip_candidate = item.get("clip", {})
                                if isinstance(clip_candidate, dict):
                                    media = clip_candidate.get("clip") if "clip" in clip_candidate else clip_candidate
                            if isinstance(media, dict):
                                code = media.get("code")
                                if code:
                                    reel_url = f"https://www.instagram.com/reel/{code}/"
                        elif item_type == "text":
                            match = REEL_REGEX.search(text_val)
                            if match:
                                reel_url = f"https://www.instagram.com/reel/{match.group(1)}/"
                        elif item_type in ("link", "xma_media_share"):
                            link_text = item.get("link", {}).get("text", "") or item.get("xma_media_share", {}).get("url", "")
                            match = REEL_REGEX.search(link_text)
                            if match:
                                reel_url = f"https://www.instagram.com/reel/{match.group(1)}/"

                        if reel_url:
                            res = await self.process_message(
                                mid=mid,
                                text=text_val,
                                sender_id=sender_id,
                                url=reel_url
                            )
                            results.append(res)
            finally:
                if close_client:
                    await client.aclose()

        except Exception as e:
            logger.warning(f"Exception during Instagram DM polling: {e}")

        return results
