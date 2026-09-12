import pytest
from unittest.mock import MagicMock
from modules.search import SearchEngine, cosine_similarity


def test_cosine_similarity():
    assert pytest.approx(cosine_similarity([1.0, 0.0], [1.0, 0.0])) == 1.0
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0])) == 0.0
    assert pytest.approx(cosine_similarity([0.0, 0.0], [1.0, 1.0])) == 0.0


def test_hybrid_search():
    mock_db = MagicMock()
    mock_analyzer = MagicMock()
    mock_analyzer.generate_embedding.return_value = [1.0, 0.0]
    
    mock_db.search_fts.return_value = [{"id": 1, "title": "Mobility Routine", "category": "workout", "url": "u1"}]
    mock_db.get_all_embeddings.return_value = [
        {"id": 1, "title": "Mobility Routine", "category": "workout", "url": "u1", "embedding": [0.95, 0.05]},
        {"id": 2, "title": "Stretching Tips", "category": "workout", "url": "u2", "embedding": [0.99, 0.01]}
    ]
    
    engine = SearchEngine(db=mock_db, analyzer=mock_analyzer)
    results = engine.hybrid_search("pre gym mobility", top_k=5)
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[1]["id"] == 2


def test_answer_conversational_query_no_matches():
    mock_db = MagicMock()
    mock_analyzer = MagicMock()
    mock_db.search_fts.return_value = []
    mock_db.get_all_embeddings.return_value = []
    mock_analyzer.generate_embedding.return_value = [1.0, 0.0]

    engine = SearchEngine(db=mock_db, analyzer=mock_analyzer)
    ans = engine.answer_conversational_query("Where is the pasta?")
    assert "couldn't find" in ans.lower()


def test_answer_conversational_query_with_matches():
    mock_db = MagicMock()
    mock_analyzer = MagicMock()
    mock_analyzer.generate_embedding.return_value = [1.0, 0.0]
    
    mock_db.search_fts.return_value = [{
        "id": 1,
        "title": "Protein Pancakes",
        "category": "recipe",
        "user_intent": "Breakfast meal prep",
        "url": "https://instagram.com/reel/C-pancake/"
    }]
    mock_db.get_all_embeddings.return_value = []
    mock_db.get_entities.return_value = [
        {"text": "Bake at 180C", "start_ts": 31.0, "confidence": 0.94}
    ]

    mock_resp = MagicMock()
    mock_resp.text = "Bake at 180°C (Evidence: 00:31, 94% conf). Source: https://instagram.com/reel/C-pancake/"
    mock_analyzer.client.models.generate_content.return_value = mock_resp

    engine = SearchEngine(db=mock_db, analyzer=mock_analyzer)
    response = engine.answer_conversational_query("What temp do I bake the pancakes at?")
    assert "180°C" in response
    assert "00:31" in response
