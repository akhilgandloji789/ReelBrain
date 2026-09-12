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
