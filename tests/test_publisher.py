import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from modules.analyzer import ReelAnalysisOutput, EntityItem
from modules.publisher import TelegramPublisher, format_timestamp


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(8.4) == "00:08"
    assert format_timestamp(75.0) == "01:15"


def test_format_note_html_truthful_confidence():
    publisher = TelegramPublisher(
        bot_token="fake_token",
        group_chat_id="-100123456",
        topic_map={"recipe": 10, "tech": 20}
    )
    analysis = ReelAnalysisOutput(
        title="Quick Oats",
        category="recipe",
        tldr="Protein breakfast",
        entities=[
            EntityItem(entity_type="ingredient", text="50g Oats", start_ts=8.0, confidence=None),
            EntityItem(entity_type="instruction", text="Add milk", start_ts=15.0, confidence=0.94)
        ],
        keyframe_timestamps=[8.0]
    )
    html = publisher.format_note_html(
        analysis=analysis,
        original_url="https://instagram.com/reel/C-test/",
        user_intent="Sunday meal prep"
    )
    
    assert "💡 <b>Why You Saved This:</b> Sunday meal prep" in html
    assert "• 50g Oats <i>(⏱️ 00:08)</i>" in html
    assert "• Add milk <i>(⏱️ 00:15)</i>" in html
    assert 'href="https://instagram.com/reel/C-test/"' in html

    # Test recipe buttons
    kb_recipe = publisher.build_inline_keyboard(reel_id=42, category="recipe")
    buttons = [[b.text for b in row] for row in kb_recipe.inline_keyboard]
    assert ["🛒 Grocery List", "🔍 Ask Memory"] in buttons
    assert ["👍 Accurate", "👎 Inaccurate"] in buttons

    # Test tech buttons
    kb_tech = publisher.build_inline_keyboard(reel_id=99, category="tech")
    tech_buttons = [[b.text for b in row] for row in kb_tech.inline_keyboard]
    assert ["💻 Copy Code", "🔍 Ask Memory"] in tech_buttons


@pytest.mark.asyncio
async def test_publish_reel_async(tmp_path: Path):
    fake_frame = tmp_path / "frame1.jpg"
    fake_frame.write_bytes(b"image")

    mock_bot = AsyncMock()
    publisher = TelegramPublisher(
        bot_token="12345:fake",
        group_chat_id=-100999,
        topic_map={"recipe": 77},
        bot=mock_bot
    )
    analysis = ReelAnalysisOutput(
        title="Pancakes",
        category="recipe",
        tldr="Fluffy pancakes",
        entities=[EntityItem(entity_type="ingredient", text="Flour", start_ts=1.0)],
        keyframe_timestamps=[1.0]
    )

    thread_id = await publisher.publish_reel(
        reel_id=1,
        analysis=analysis,
        frame_paths=[fake_frame],
        original_url="https://instagram.com/reel/C-abc/",
        user_intent="Breakfast idea"
    )
    assert thread_id == 77
    mock_bot.send_media_group.assert_awaited_once()
    mock_bot.send_message.assert_awaited_once()
