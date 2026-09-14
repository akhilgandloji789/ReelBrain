import pytest
from pathlib import Path
from modules.storage import ReelDatabase, pack_vector, unpack_vector

def test_database_intent_and_fts_sync(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    shortcode = "C-xyz123"
    url = f"https://www.instagram.com/reel/{shortcode}/"
    
    reel_id = db.add_reel(
        url=url,
        shortcode=shortcode,
        category="recipe",
        title="10-Minute High-Protein Oats",
        user_intent="Try this for Sunday meal prep",
        raw_transcript="Oats and whey recipe"
    )
    
    reel = db.get_reel_by_id(reel_id)
    assert reel is not None
    assert reel["user_intent"] == "Try this for Sunday meal prep"
    assert reel["status"] == "RECEIVED"
    
    db.update_reel_status(reel_id, "COMPLETED")
    assert db.is_processed(shortcode) is True

    # Entities with nullable confidence (no fake default)
    entities = [
        {"entity_type": "ingredient", "text": "50g rolled oats", "start_ts": 8.0, "confidence": None},
        {"entity_type": "instruction", "text": "Mix with almond milk", "start_ts": 15.0, "confidence": 0.94}
    ]
    db.add_entities(reel_id, entities)
    
    stored = db.get_entities(reel_id)
    assert len(stored) == 2
    assert stored[0]["confidence"] is None
    assert stored[1]["confidence"] == 0.94

    # Search user intent via FTS5
    results = db.search_fts("meal prep")
    assert len(results) == 1
    assert results[0]["id"] == reel_id

    # Search entity text via FTS5
    results_ingr = db.search_fts("oats")
    assert len(results_ingr) == 1
    assert results_ingr[0]["id"] == reel_id

def test_database_entity_update_and_fts_reindex(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    reel_id = db.add_reel(
        url="https://www.instagram.com/reel/C-tech456/",
        shortcode="C-tech456",
        category="tech",
        title="Docker Tips",
        user_intent="Study container caching",
        raw_transcript="Multi-stage docker build tutorial"
    )
    db.add_entities(reel_id, [
        {"entity_type": "code", "text": "RUN apt-get update", "start_ts": 5.0, "confidence": 0.99}
    ])
    
    ents = db.get_entities(reel_id)
    entity_id = ents[0]["id"]
    
    # Update entity
    success, r_id = db.update_entity(entity_id, "RUN apt-get update && apt-get install -y curl")
    assert success is True
    assert r_id == reel_id
    
    updated_ents = db.get_entities(reel_id)
    assert updated_ents[0]["text"] == "RUN apt-get update && apt-get install -y curl"
    assert updated_ents[0]["edited_by_user"] == 1
    
    # Verify FTS index includes the newly edited text
    fts_results = db.search_fts("curl")
    assert len(fts_results) == 1
    assert fts_results[0]["id"] == reel_id

def test_embeddings_pack_unpack_and_storage(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    reel_id = db.add_reel(
        url="https://www.instagram.com/reel/C-emb789/",
        shortcode="C-emb789",
        category="idea",
        title="Atomic Habits",
        user_intent="Habit stacking ideas",
        raw_transcript="Build small habits everyday"
    )
    
    vec = [0.12, -0.34, 0.56, 0.78]
    blob = pack_vector(vec)
    unpacked = unpack_vector(blob)
    assert pytest.approx(vec) == unpacked
    
    db.store_embedding(reel_id=reel_id, model="models/gemini-embedding-001", dim=4, vector=vec)
    
    all_embeddings = db.get_all_embeddings()
    assert len(all_embeddings) == 1
    assert all_embeddings[0]["id"] == reel_id
    assert pytest.approx(all_embeddings[0]["embedding"]) == vec

def test_action_logging_and_metrics(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    reel_id = db.add_reel(
        url="https://www.instagram.com/reel/C-action1/",
        shortcode="C-action1",
        category="recipe",
        title="Pancakes",
        user_intent="Breakfast idea",
        raw_transcript=""
    )
    db.update_reel_status(reel_id, "COMPLETED")
    db.log_action(reel_id, "grocery", {"ingredients": ["flour", "milk"]})
    
    metrics = db.get_metrics_summary()
    assert metrics["total_reels"] == 1
    assert metrics["total_actions"] == 1

def test_cleanup_orphaned_jobs(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    r1 = db.add_reel("https://www.instagram.com/reel/C-old1/", "C-old1", "other", "Stuck 1")
    db.update_reel_status(r1, "DOWNLOADING")
    
    # Fake backdate the saved_at
    with db._get_connection() as conn:
        conn.execute("UPDATE reels SET saved_at = '2020-01-01T00:00:00' WHERE id = ?", (r1,))
        conn.commit()
        
    cleaned = db.cleanup_orphaned_jobs(max_age_hours=1)
    assert cleaned == 1
    
    stuck_reel = db.get_reel_by_id(r1)
    assert stuck_reel["status"] == "FAILED"
    assert "Timeout" in (stuck_reel["error_message"] or "")


def test_favorites_storage_and_toggle(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_reelminds.db")
    r1 = db.add_reel("https://www.instagram.com/reel/C-fav1/", "C-fav1", "recipe", "Fav Pasta")
    db.update_reel_status(r1, "COMPLETED")

    # Not favorited initially
    assert db.is_favorite(r1) is False
    favs, count = db.get_favorites()
    assert count == 0

    # Toggle to favorite
    now_fav = db.toggle_favorite(r1)
    assert now_fav is True
    assert db.is_favorite(r1) is True

    favs, count = db.get_favorites()
    assert count == 1
    assert favs[0]["id"] == r1
    assert favs[0]["title"] == "Fav Pasta"

    # Toggle again to remove favorite
    now_fav = db.toggle_favorite(r1)
    assert now_fav is False
    assert db.is_favorite(r1) is False

    favs, count = db.get_favorites()
    assert count == 0


def test_reels_by_period_and_highlights(tmp_path: Path):
    from datetime import datetime, timedelta, timezone
    db = ReelDatabase(tmp_path / "test_reelminds.db")

    r_today = db.add_reel("https://www.instagram.com/reel/C-today/", "C-today", "tech", "AI Agent")
    db.update_reel_status(r_today, "COMPLETED")

    r_week = db.add_reel("https://www.instagram.com/reel/C-week/", "C-week", "recipe", "Weekly Cake")
    db.update_reel_status(r_week, "COMPLETED")
    with db._get_connection() as conn:
        three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        conn.execute("UPDATE reels SET saved_at = ? WHERE id = ?", (three_days_ago, r_week))
        conn.commit()

    r_old = db.add_reel("https://www.instagram.com/reel/C-old/", "C-old", "workout", "Old Leg Day")
    db.update_reel_status(r_old, "COMPLETED")
    with db._get_connection() as conn:
        two_months_ago = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        conn.execute("UPDATE reels SET saved_at = ? WHERE id = ?", (two_months_ago, r_old))
        conn.commit()

    # Today should only have r_today
    today_reels = db.get_reels_by_period("today")
    assert len(today_reels) == 1
    assert today_reels[0]["id"] == r_today

    # Week should have r_today and r_week
    week_reels = db.get_reels_by_period("week")
    assert len(week_reels) == 2
    assert {r["id"] for r in week_reels} == {r_today, r_week}

    # Month should have r_today and r_week (not r_old)
    month_reels = db.get_reels_by_period("month")
    assert len(month_reels) == 2

    # Test category counts by period
    counts_today = db.get_category_counts_by_period("today")
    assert counts_today.get("tech") == 1
    assert counts_today.get("recipe", 0) == 0

    counts_week = db.get_category_counts_by_period("week")
    assert counts_week.get("tech") == 1
    assert counts_week.get("recipe") == 1
