import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.analyzer import ReelAnalysisOutput, EntityItem, ReelAnalyzer


def test_entity_nullable_confidence():
    item = EntityItem(entity_type="ingredient", text="50g Oats", start_ts=5.0)
    assert item.confidence is None  # Nullable, no fake 0.90 default!

    item_with_conf = EntityItem(entity_type="ingredient", text="50g Oats", start_ts=5.0, confidence=0.96)
    assert item_with_conf.confidence == 0.96


def test_reel_analysis_output_schema():
    data = {
        "title": "High Protein Oats",
        "category": "recipe",
        "tldr": "Quick healthy oats breakfast.",
        "entities": [
            {"entity_type": "ingredient", "text": "50g rolled oats", "start_ts": 2.5, "confidence": None},
            {"entity_type": "instruction", "text": "Boil almond milk", "start_ts": 8.0, "confidence": 0.95}
        ],
        "keyframe_timestamps": [2.5, 8.0]
    }
    output = ReelAnalysisOutput.model_validate(data)
    assert output.category == "recipe"
    assert len(output.entities) == 2
    assert output.entities[0].confidence is None
    assert output.entities[1].confidence == 0.95


def test_generate_embedding_dim_slicing():
    with patch("modules.analyzer.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_embed_resp = MagicMock()
        # Suppose model returns 1000 dims
        mock_embed_resp.embedding.values = [0.1] * 1000
        mock_client.models.embed_content.return_value = mock_embed_resp

        analyzer = ReelAnalyzer(api_key="dummy_key", embedding_dim=768)
        emb = analyzer.generate_embedding("some text")
        assert len(emb) == 768


def test_analyze_video_mocked(tmp_path: Path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"dummy video data")

    with patch("modules.analyzer.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_uploaded = MagicMock()
        mock_uploaded.state = "ACTIVE"
        mock_uploaded.name = "files/test123"
        mock_client.files.upload.return_value = mock_uploaded

        mock_resp = MagicMock()
        mock_resp.text = """{
            "title": "Python Asyncio Tips",
            "category": "tech",
            "tldr": "Learn how to use asyncio TaskGroups.",
            "entities": [
                {"entity_type": "code", "text": "async with asyncio.TaskGroup() as tg:", "start_ts": 3.0, "confidence": 0.98}
            ],
            "keyframe_timestamps": [3.0]
        }"""
        mock_client.models.generate_content.return_value = mock_resp

        analyzer = ReelAnalyzer(api_key="dummy_key")
        res = analyzer.analyze_video(video)
        assert res.title == "Python Asyncio Tips"
        assert res.category == "tech"
        assert len(res.entities) == 1
        assert res.entities[0].confidence == 0.98
        mock_client.files.delete.assert_called_once_with(name="files/test123")
