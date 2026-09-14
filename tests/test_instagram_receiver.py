import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from modules.storage import ReelDatabase
from modules.instagram_receiver import InstagramReceiver


@pytest.fixture
def receiver_setup(tmp_path: Path):
    db = ReelDatabase(tmp_path / "test_receiver.db")
    mock_pipeline = MagicMock()
    mock_pipeline.process_url = AsyncMock(return_value=(True, "Indexed Reel", 10))
    receiver = InstagramReceiver(
        pipeline=mock_pipeline,
        db=db,
        session_id="dummy_session_123",
        verify_token="meta_secret_verify_token"
    )
    return receiver, db, mock_pipeline


def test_verify_webhook(receiver_setup):
    receiver, _, _ = receiver_setup

    # Valid handshake
    challenge = receiver.verify_webhook(
        mode="subscribe",
        token="meta_secret_verify_token",
        challenge="CHALLENGE_STRING_123"
    )
    assert challenge == "CHALLENGE_STRING_123"

    # Invalid token
    assert receiver.verify_webhook("subscribe", "wrong_token", "challenge") is None

    # Invalid mode
    assert receiver.verify_webhook("publish", "meta_secret_verify_token", "challenge") is None


def test_extract_messages_from_webhook(receiver_setup):
    receiver, _, _ = receiver_setup

    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": "ig_page_1",
                "time": 1700000000,
                "messaging": [
                    {
                        "sender": {"id": "sender_101"},
                        "message": {
                            "mid": "mid_101",
                            "text": "https://www.instagram.com/reel/C-textReel/ check out this recipe"
                        }
                    },
                    {
                        "sender": {"id": "sender_102"},
                        "message": {
                            "mid": "mid_102",
                            "attachments": [
                                {
                                    "type": "share",
                                    "payload": {
                                        "url": "https://www.instagram.com/reel/C-shareReel/?igsh=xyz"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    messages = receiver.extract_messages_from_webhook(payload)
    assert len(messages) == 2
    assert messages[0]["mid"] == "mid_101"
    assert messages[0]["url"] == "https://www.instagram.com/reel/C-textReel/"
    assert "check out this recipe" in (messages[0]["intent"] or "")

    assert messages[1]["mid"] == "mid_102"
    assert messages[1]["url"] == "https://www.instagram.com/reel/C-shareReel/"


@pytest.mark.asyncio
async def test_process_message_deduplication(receiver_setup):
    receiver, db, mock_pipeline = receiver_setup

    mid = "unique_mid_456"
    text = "https://www.instagram.com/reel/C-unique456/ Sunday meal prep"

    # First delivery
    success, msg, thread_id = await receiver.process_message(
        mid=mid,
        text=text,
        sender_id="sender_1"
    )
    assert success is True
    assert db.is_dm_processed(mid) is True
    mock_pipeline.process_url.assert_awaited_once()
    assert "https://www.instagram.com/reel/C-unique456/" in mock_pipeline.process_url.call_args[0][0]

    # Duplicate delivery with same mid
    mock_pipeline.process_url.reset_mock()
    success2, msg2, thread_id2 = await receiver.process_message(
        mid=mid,
        text=text,
        sender_id="sender_1"
    )
    assert success2 is False
    assert "already processed" in msg2
    mock_pipeline.process_url.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_webhook_full_payload(receiver_setup):
    receiver, _, mock_pipeline = receiver_setup

    payload = {
        "object": "instagram",
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": "user_abc"},
                        "message": {
                            "mid": "mid_batch_1",
                            "text": "https://www.instagram.com/reel/C-batch1/"
                        }
                    }
                ]
            }
        ]
    }

    results = await receiver.handle_webhook(payload)
    assert len(results) == 1
    assert results[0]["mid"] == "mid_batch_1"
    assert results[0]["success"] is True


@pytest.mark.asyncio
async def test_poll_direct_inbox_mocked(receiver_setup):
    receiver, db, mock_pipeline = receiver_setup

    mock_inbox_data = {
        "inbox": {
            "threads": [
                {
                    "items": [
                        {
                            "item_id": "item_share_1",
                            "user_id": "999",
                            "item_type": "media_share",
                            "media": {"code": "C-inboxReel1"}
                        },
                        {
                            "item_id": "item_text_2",
                            "user_id": "888",
                            "item_type": "text",
                            "text": "Check this: https://www.instagram.com/reel/C-inboxReel2/ high protein"
                        }
                    ]
                }
            ]
        }
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_inbox_data

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.aclose = AsyncMock()

    receiver._http_client = mock_client

    results = await receiver.poll_direct_inbox()
    assert len(results) == 2
    assert db.is_dm_processed("item_share_1") is True
    assert db.is_dm_processed("item_text_2") is True
    assert mock_pipeline.process_url.await_count == 2


def test_extract_messages_from_webhook_attachment_with_comment(receiver_setup):
    receiver, _, _ = receiver_setup

    payload = {
        "object": "instagram",
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": "sender_caption"},
                        "message": {
                            "mid": "mid_with_comment",
                            "text": "Try making this for dinner tomorrow!",
                            "attachments": [
                                {
                                    "type": "share",
                                    "payload": {
                                        "url": "https://www.instagram.com/reel/C-captionReel/"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    messages = receiver.extract_messages_from_webhook(payload)
    assert len(messages) == 1
    assert messages[0]["mid"] == "mid_with_comment"
    assert messages[0]["url"] == "https://www.instagram.com/reel/C-captionReel/"
    # Verify the user's comment is properly preserved as the intent!
    assert messages[0]["intent"] == "Try making this for dinner tomorrow!"


@pytest.mark.asyncio
async def test_process_message_empty_mid_deterministic_hash(receiver_setup):
    receiver, db, mock_pipeline = receiver_setup

    # When mid is empty, it should generate a stable hash and deduplicate
    success1, msg1, _ = await receiver.process_message(
        mid="",
        text="",
        sender_id="user_hash_test",
        url="https://www.instagram.com/reel/C-hashReel/"
    )
    assert success1 is True

    # Duplicate with empty mid
    mock_pipeline.process_url.reset_mock()
    success2, msg2, _ = await receiver.process_message(
        mid="",
        text="",
        sender_id="user_hash_test",
        url="https://www.instagram.com/reel/C-hashReel/"
    )
    assert success2 is False
    assert "already processed" in msg2


@pytest.mark.asyncio
async def test_poll_direct_inbox_nested_clip_and_felix(receiver_setup):
    receiver, db, mock_pipeline = receiver_setup

    mock_inbox_data = {
        "inbox": {
            "threads": [
                {
                    "items": [
                        {
                            "item_id": "clip_nested_1",
                            "user_id": "111",
                            "item_type": "clip",
                            "clip": {"clip": {"code": "C-nestedClip"}}
                        },
                        {
                            "item_id": "felix_share_2",
                            "user_id": "222",
                            "item_type": "felix_share",
                            "media": {"code": "C-felixReel"}
                        }
                    ]
                }
            ]
        }
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_inbox_data

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.aclose = AsyncMock()
    receiver._http_client = mock_client

    results = await receiver.poll_direct_inbox()
    assert len(results) == 2
    assert db.is_dm_processed("clip_nested_1") is True
    assert db.is_dm_processed("felix_share_2") is True
