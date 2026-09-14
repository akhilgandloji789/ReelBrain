import json
import sqlite3
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def clean_handle(handle: str) -> str:
    h = handle.strip()
    if "/" in h:
        h = h.split("?")[0].rstrip("/")
        parts = [p for p in h.split("/") if p and "instagram.com" not in p and "instagr.am" not in p and "http" not in p]
        if parts:
            h = parts[-1]
    return h.lstrip("@").lower()


CATEGORY_ALIASES: dict[str, str] = {
    "recipe": "recipe",
    "recipes": "recipe",
    "food": "recipe",
    "cooking": "recipe",
    "tech": "tech",
    "technology": "tech",
    "ai": "tech",
    "code": "tech",
    "coding": "tech",
    "workout": "workout",
    "fitness": "workout",
    "gym": "workout",
    "exercise": "workout",
    "idea": "idea",
    "ideas": "idea",
    "book": "idea",
    "books": "idea",
    "travel": "travel",
    "trip": "travel",
    "finance": "finance",
    "money": "finance",
    "other": "other",
    "all": "all",
}


class ReelDatabase:
    def __init__(self, db_path: Path | str = Path("data/reelminds.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reels (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    url             TEXT UNIQUE NOT NULL,
                    shortcode       TEXT UNIQUE NOT NULL,
                    content_hash    TEXT,
                    saved_at        TEXT NOT NULL,
                    category        TEXT NOT NULL,
                    title           TEXT,
                    user_intent     TEXT,
                    raw_transcript  TEXT,
                    status          TEXT NOT NULL DEFAULT 'RECEIVED',
                    error_message   TEXT
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    reel_id         INTEGER NOT NULL REFERENCES reels(id) ON DELETE CASCADE,
                    entity_type     TEXT NOT NULL,
                    text            TEXT NOT NULL,
                    start_ts        REAL NOT NULL,
                    end_ts          REAL,
                    confidence      REAL,
                    frame_path      TEXT,
                    edited_by_user  INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    reel_id     INTEGER PRIMARY KEY REFERENCES reels(id) ON DELETE CASCADE,
                    model       TEXT NOT NULL,
                    dim         INTEGER NOT NULL,
                    vector      BLOB NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS reels_fts USING fts5(
                    title, user_intent, raw_transcript, entity_text,
                    tokenize='porter unicode61'
                );

                CREATE TABLE IF NOT EXISTS action_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    reel_id     INTEGER REFERENCES reels(id),
                    action_type TEXT NOT NULL,
                    payload     TEXT,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tracked_channels (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    handle          TEXT UNIQUE NOT NULL,
                    added_at        TEXT NOT NULL,
                    last_checked_at TEXT,
                    last_shortcode  TEXT,
                    is_active       INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS processed_dms (
                    mid             TEXT PRIMARY KEY,
                    sender_id       TEXT,
                    url             TEXT,
                    processed_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS favorites (
                    reel_id         INTEGER PRIMARY KEY REFERENCES reels(id) ON DELETE CASCADE,
                    favorited_at    TEXT NOT NULL
                );
            """)
            conn.commit()

    def is_processed(self, shortcode: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM reels WHERE shortcode = ? AND status = 'COMPLETED'",
                (shortcode,)
            )
            return cursor.fetchone() is not None

    def add_reel(
        self,
        url: str,
        shortcode: str,
        category: str,
        title: str,
        user_intent: str | None = None,
        raw_transcript: str = "",
        content_hash: str | None = None
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO reels (url, shortcode, content_hash, saved_at, category, title, user_intent, raw_transcript, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RECEIVED')
                """,
                (url, shortcode, content_hash, now, category, title, user_intent, raw_transcript)
            )
            reel_id = cursor.lastrowid
            self._sync_fts_for_reel(conn, reel_id)
            conn.commit()
            return reel_id

    def update_reel_status(self, reel_id: int, status: str, error_message: str | None = None) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE reels SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, reel_id)
            )
            conn.commit()

    def update_reel_metadata(self, reel_id: int, category: str, title: str, raw_transcript: str = "") -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE reels SET category = ?, title = ?, raw_transcript = ? WHERE id = ?",
                (category, title, raw_transcript, reel_id)
            )
            self._sync_fts_for_reel(conn, reel_id)
            conn.commit()

    def add_entities(self, reel_id: int, entities: list[dict[str, Any]]) -> None:
        with self._get_connection() as conn:
            for ent in entities:
                conn.execute(
                    """
                    INSERT INTO entities (reel_id, entity_type, text, start_ts, end_ts, confidence, frame_path, edited_by_user)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reel_id,
                        ent.get("entity_type", "fact"),
                        ent.get("text", ""),
                        float(ent.get("start_ts", 0.0)),
                        float(ent["end_ts"]) if ent.get("end_ts") is not None else None,
                        float(ent["confidence"]) if ent.get("confidence") is not None else None,
                        ent.get("frame_path"),
                        int(ent.get("edited_by_user", 0))
                    )
                )
            self._sync_fts_for_reel(conn, reel_id)
            conn.commit()

    def _sync_fts_for_reel(self, conn: sqlite3.Connection, reel_id: int) -> None:
        reel = conn.execute("SELECT title, user_intent, raw_transcript FROM reels WHERE id = ?", (reel_id,)).fetchone()
        if not reel:
            return
        ents = conn.execute("SELECT text FROM entities WHERE reel_id = ?", (reel_id,)).fetchall()
        all_entity_text = " ".join([e["text"] for e in ents])

        conn.execute("DELETE FROM reels_fts WHERE rowid = ?", (reel_id,))
        conn.execute(
            "INSERT INTO reels_fts(rowid, title, user_intent, raw_transcript, entity_text) VALUES (?, ?, ?, ?, ?)",
            (reel_id, reel["title"] or "", reel["user_intent"] or "", reel["raw_transcript"] or "", all_entity_text)
        )

    def get_reel_by_shortcode(self, shortcode: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM reels WHERE shortcode = ?", (shortcode,)).fetchone()
            return dict(row) if row else None

    def get_reel_by_id(self, reel_id: int) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM reels WHERE id = ?", (reel_id,)).fetchone()
            return dict(row) if row else None

    def get_entities(self, reel_id: int) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM entities WHERE reel_id = ? ORDER BY start_ts ASC", (reel_id,))
            return [dict(row) for row in cursor.fetchall()]

    def update_entity(self, entity_id: int, new_text: str) -> tuple[bool, int]:
        with self._get_connection() as conn:
            ent = conn.execute("SELECT reel_id FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if not ent:
                return False, 0
            reel_id = ent["reel_id"]
            conn.execute("UPDATE entities SET text = ?, edited_by_user = 1 WHERE id = ?", (new_text, entity_id))
            self._sync_fts_for_reel(conn, reel_id)
            conn.commit()
            return True, reel_id

    def store_embedding(self, reel_id: int, model: str, dim: int, vector: list[float]) -> None:
        blob = pack_vector(vector)
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (reel_id, model, dim, vector, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (reel_id, model, dim, blob, now)
            )
            conn.commit()

    def get_all_embeddings(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.id, r.title, r.category, r.url, r.user_intent, e.model, e.dim, e.vector
                FROM embeddings e
                JOIN reels r ON e.reel_id = r.id
                """
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["embedding"] = unpack_vector(d["vector"])
                results.append(d)
            return results

    def search_fts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        clean_words = [w for w in "".join(c if c.isalnum() else " " for c in query).split() if w]
        if not clean_words:
            return []
        # FTS5 formatted phrase query
        fts_query = " ".join(f'"{w}"' for w in clean_words)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.* FROM reels r
                JOIN reels_fts ON r.id = reels_fts.rowid
                WHERE reels_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def log_action(self, reel_id: int | None, action_type: str, payload: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO action_log (reel_id, action_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (reel_id, action_type, json.dumps(payload), now)
            )
            conn.commit()

    def get_recent_reels(self, days: int = 7) -> list[dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM reels WHERE saved_at >= ? ORDER BY saved_at DESC", (since,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_reels_with_entities(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            reels = [dict(r) for r in conn.execute("SELECT * FROM reels ORDER BY saved_at DESC").fetchall()]
            for r in reels:
                ents = conn.execute("SELECT * FROM entities WHERE reel_id = ? ORDER BY start_ts ASC", (r["id"],)).fetchall()
                r["entities"] = [dict(e) for e in ents]
            return reels

    def get_metrics_summary(self) -> dict[str, int]:
        with self._get_connection() as conn:
            total_reels = conn.execute("SELECT COUNT(*) FROM reels WHERE status = 'COMPLETED'").fetchone()[0]
            total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            total_actions = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]
            return {
                "total_reels": total_reels,
                "total_entities": total_entities,
                "total_embeddings": total_embeddings,
                "total_actions": total_actions
            }

    def cleanup_orphaned_jobs(self, max_age_hours: int = 1) -> int:
        threshold = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE reels
                SET status = 'FAILED', error_message = 'Timeout: orphaned job during startup recovery'
                WHERE status NOT IN ('COMPLETED', 'FAILED') AND saved_at <= ?
                """,
                (threshold,)
            )
            conn.commit()
            return cursor.rowcount

    def add_tracked_channel(self, handle: str) -> bool:
        clean = clean_handle(handle)
        if not clean:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO tracked_channels (handle, added_at, is_active)
                VALUES (?, ?, 1)
                ON CONFLICT(handle) DO UPDATE SET is_active = 1
                """,
                (clean, now)
            )
            conn.commit()
            return True

    def remove_tracked_channel(self, handle: str) -> bool:
        clean = clean_handle(handle)
        if not clean:
            return False
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE tracked_channels SET is_active = 0 WHERE handle = ? AND is_active = 1",
                (clean,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_tracked_channels(self, active_only: bool = True) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            if active_only:
                cursor = conn.execute("SELECT * FROM tracked_channels WHERE is_active = 1 ORDER BY handle ASC")
            else:
                cursor = conn.execute("SELECT * FROM tracked_channels ORDER BY handle ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_tracked_channel(self, handle: str) -> dict[str, Any] | None:
        clean = clean_handle(handle)
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM tracked_channels WHERE handle = ?", (clean,)).fetchone()
            return dict(row) if row else None

    def update_channel_last_checked(self, handle: str, last_shortcode: str | None = None) -> None:
        clean = clean_handle(handle)
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            if last_shortcode:
                conn.execute(
                    "UPDATE tracked_channels SET last_checked_at = ?, last_shortcode = ? WHERE handle = ?",
                    (now, last_shortcode, clean)
                )
            else:
                conn.execute(
                    "UPDATE tracked_channels SET last_checked_at = ? WHERE handle = ?",
                    (now, clean)
                )
            conn.commit()

    def is_dm_processed(self, mid: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM processed_dms WHERE mid = ?", (mid,))
            return cursor.fetchone() is not None

    def record_processed_dm(self, mid: str, sender_id: str | None = None, url: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO processed_dms (mid, sender_id, url, processed_at) VALUES (?, ?, ?, ?)",
                (mid, sender_id, url, now)
            )
            conn.commit()

    def get_category_counts(self) -> dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT LOWER(category) as category, COUNT(*) as count FROM reels WHERE status = 'COMPLETED' GROUP BY LOWER(category)"
            )
            return {row["category"]: row["count"] for row in cursor.fetchall()}

    def get_reels_by_category(self, category: str | None = None, limit: int = 5, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        with self._get_connection() as conn:
            raw_cat = (category or "").strip().lstrip("#").lower()
            cat_clean = CATEGORY_ALIASES.get(raw_cat, raw_cat)
            if not cat_clean or cat_clean == "all":
                total = conn.execute("SELECT COUNT(*) FROM reels WHERE status = 'COMPLETED'").fetchone()[0]
                cursor = conn.execute(
                    "SELECT * FROM reels WHERE status = 'COMPLETED' ORDER BY saved_at DESC, id DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )
            else:
                total = conn.execute(
                    "SELECT COUNT(*) FROM reels WHERE status = 'COMPLETED' AND LOWER(category) = ?",
                    (cat_clean,)
                ).fetchone()[0]
                cursor = conn.execute(
                    "SELECT * FROM reels WHERE status = 'COMPLETED' AND LOWER(category) = ? ORDER BY saved_at DESC, id DESC LIMIT ? OFFSET ?",
                    (cat_clean, limit, offset)
                )
            return [dict(row) for row in cursor.fetchall()], total

    def is_favorite(self, reel_id: int) -> bool:
        with self._get_connection() as conn:
            row = conn.execute("SELECT 1 FROM favorites WHERE reel_id = ?", (reel_id,)).fetchone()
            return row is not None

    def add_favorite(self, reel_id: int) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO favorites (reel_id, favorited_at) VALUES (?, ?)",
                (reel_id, now)
            )
            conn.commit()
            return True

    def remove_favorite(self, reel_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM favorites WHERE reel_id = ?", (reel_id,))
            conn.commit()
            return cursor.rowcount > 0

    def toggle_favorite(self, reel_id: int) -> bool:
        if self.is_favorite(reel_id):
            self.remove_favorite(reel_id)
            return False
        else:
            self.add_favorite(reel_id)
            return True

    def get_favorites(self, limit: int = 20, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        with self._get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM favorites f JOIN reels r ON f.reel_id = r.id WHERE r.status = 'COMPLETED'"
            ).fetchone()[0]
            cursor = conn.execute(
                """
                SELECT r.*, f.favorited_at
                FROM favorites f
                JOIN reels r ON f.reel_id = r.id
                WHERE r.status = 'COMPLETED'
                ORDER BY f.favorited_at DESC, r.id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            return [dict(row) for row in cursor.fetchall()], total

    def get_reels_by_period(self, period: str = "today") -> list[dict[str, Any]]:
        period_clean = period.lower().strip()
        now = datetime.now(timezone.utc)
        if period_clean in ("today", "day", "24h"):
            since = (now - timedelta(days=1)).isoformat()
        elif period_clean in ("week", "7d"):
            since = (now - timedelta(days=7)).isoformat()
        elif period_clean in ("month", "30d"):
            since = (now - timedelta(days=30)).isoformat()
        else:
            since = (now - timedelta(days=1)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM reels
                WHERE status = 'COMPLETED' AND saved_at >= ?
                ORDER BY saved_at DESC, id DESC
                """,
                (since,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_category_counts_by_period(self, period: str = "today") -> dict[str, int]:
        reels = self.get_reels_by_period(period)
        counts: dict[str, int] = {}
        for r in reels:
            cat = (r.get("category") or "other").lower()
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def get_all_week_reels_with_entities(self, days: int = 7) -> list[dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._get_connection() as conn:
            reels = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM reels WHERE status = 'COMPLETED' AND saved_at >= ? ORDER BY saved_at DESC, id DESC",
                    (since,)
                ).fetchall()
            ]
            for r in reels:
                ents = conn.execute(
                    "SELECT * FROM entities WHERE reel_id = ? ORDER BY start_ts ASC",
                    (r["id"],)
                ).fetchall()
                r["entities"] = [dict(e) for e in ents]
            return reels
