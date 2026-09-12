import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from modules.actions import ActionHandler


def test_grocery_and_feedback():
    mock_db = MagicMock()
    mock_db.get_entities.return_value = [
        {"entity_type": "ingredient", "text": "50g Oats"},
        {"entity_type": "ingredient", "text": "1 cup milk"}
    ]
    handler = ActionHandler(db=mock_db, analyzer=MagicMock())
    
    grocery = handler.generate_grocery_list(reel_id=1)
    assert "• [ ] 50g Oats" in grocery
    assert "• [ ] 1 cup milk" in grocery
    mock_db.log_action.assert_called_with(1, "grocery", {"items": ["50g Oats", "1 cup milk"]})
    
    msg_up = handler.record_feedback(reel_id=1, thumb="up")
    assert "Thanks for the feedback" in msg_up
    mock_db.log_action.assert_called_with(1, "feedback_thumb_up", {"vote": "up"})

    msg_down = handler.record_feedback(reel_id=1, thumb="down")
    assert "inaccurate" in msg_down.lower()
    mock_db.log_action.assert_called_with(1, "feedback_thumb_down", {"vote": "down"})


def test_code_snippet_generation():
    mock_db = MagicMock()
    mock_db.get_entities.return_value = [
        {"entity_type": "code_snippet", "text": "npm install pydantic"}
    ]
    handler = ActionHandler(db=mock_db, analyzer=MagicMock())
    code_text = handler.generate_code_block(reel_id=5)
    assert "<code>npm install pydantic</code>" in code_text


def test_apply_edit_re_embeds():
    mock_db = MagicMock()
    mock_db.update_entity.return_value = (True, 10)
    mock_db.get_reel_by_id.return_value = {
        "id": 10,
        "title": "Clean Architecture",
        "user_intent": "Study design patterns"
    }
    mock_db.get_entities.return_value = [
        {"entity_type": "fact", "text": "Separate domain from IO"}
    ]

    mock_analyzer = MagicMock()
    mock_analyzer.generate_embedding.return_value = [0.5, 0.5]
    mock_analyzer.embedding_model = "models/gemini-embedding-001"
    mock_analyzer.embedding_dim = 2

    handler = ActionHandler(db=mock_db, analyzer=mock_analyzer)
    ok, msg = handler.apply_edit(entity_id=99, new_text="Separate domain from UI and IO")
    assert ok is True
    assert "refreshed" in msg
    mock_db.update_entity.assert_called_once_with(99, "Separate domain from UI and IO")
    mock_analyzer.generate_embedding.assert_called_once()
    mock_db.store_embedding.assert_called_once()


def test_export_markdown_and_json(tmp_path: Path):
    mock_db = MagicMock()
    mock_db.get_all_reels_with_entities.return_value = [{
        "id": 1,
        "title": "Avocado Toast",
        "category": "recipe",
        "user_intent": "Healthy breakfast",
        "url": "https://instagram.com/reel/C-avo/",
        "raw_transcript": "Toast the sourdough bread",
        "entities": [
            {"entity_type": "ingredient", "text": "1 avocado", "start_ts": 5.0, "confidence": 0.95}
        ]
    }]

    handler = ActionHandler(db=mock_db, analyzer=MagicMock())
    
    # Export Markdown
    md_file = tmp_path / "export.md"
    res_md = handler.export_markdown(md_file)
    assert res_md.exists()
    content_md = res_md.read_text(encoding="utf-8")
    assert "# ReelMind Knowledge Export" in content_md
    assert "Avocado Toast" in content_md
    assert "Healthy breakfast" in content_md
    assert "1 avocado" in content_md

    # Export JSON
    json_file = tmp_path / "export.json"
    res_json = handler.export_json(json_file)
    assert res_json.exists()
    data = json.loads(res_json.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["title"] == "Avocado Toast"
