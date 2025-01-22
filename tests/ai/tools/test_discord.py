"""Unit tests for Discord tools.

This module contains unit tests for the Discord toolkit functionality.
Tests are based on the working implementation from fn_test.py.
"""

import pytest
from unittest.mock import AsyncMock, patch
import json
from pathlib import Path
from datetime import datetime, timezone

from alchemist.ai.tools import DiscordTools

# Load test data from config
project_root = Path(__file__).parent.parent.parent.parent
config_path = project_root / "config" / "channels.json"
with open(config_path) as f:
    data = json.load(f)
    TEST_CHANNELS = data["channels"]
    TEST_CATEGORIES = data["categories"]

@pytest.fixture
def discord_tools():
    """Create a Discord tools instance with test data."""
    return DiscordTools(
        channels=TEST_CHANNELS,
        categories=TEST_CATEGORIES,
        service_url="http://localhost:5000"
    )

@pytest.mark.asyncio
async def test_read_channel_success(discord_tools):
    """Test successful reading of messages from a Discord channel."""
    # Mock response data based on actual format
    mock_messages = [
        {
            "id": "123456789",
            "content": "Test message content",
            "author": {
                "name": "CYBOORG",
                "id": "987654321"
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "embeds": [
                {
                    "title": "AI News Update",
                    "description": "Latest developments in AI",
                    "fields": [],
                    "color": None,
                    "footer": None,
                    "thumbnail": None,
                    "url": None
                }
            ],
            "attachments": []
        }
    ]

    # Setup mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"messages": mock_messages})

    # Setup session mock with proper async context managers
    mock_session = AsyncMock()
    mock_session.get = AsyncMock()
    mock_session.get.return_value = mock_response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        messages = await discord_tools.read_channel(
            channel_name="ai-news",
            limit=5
        )
        
        # Verify response format matches our processing
        assert len(messages) == 1
        msg = messages[0]
        assert msg["id"] == "123456789"
        assert msg["content"] == "Test message content"
        assert msg["author"] == "CYBOORG"  # Verify author name extraction
        assert len(msg["embeds"]) == 1
        assert msg["embeds"][0]["title"] == "AI News Update"
        assert msg["embeds"][0]["description"] == "Latest developments in AI"

@pytest.mark.asyncio
async def test_read_channel_with_after(discord_tools):
    """Test reading messages with timestamp filter."""
    after_time = datetime.now(timezone.utc)
    
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"messages": []})

    # Setup session mock with proper async context managers
    mock_session = AsyncMock()
    mock_session.get = AsyncMock()
    mock_session.get.return_value = mock_response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await discord_tools.read_channel(
            channel_name="ai-news",
            after=after_time,
            limit=5
        )
        
        # Verify after parameter was included in URL
        call_args = mock_session.get.call_args[0][0]
        assert "after=" in call_args
        assert after_time.isoformat() in call_args

@pytest.mark.asyncio
async def test_channel_not_found(discord_tools):
    """Test error handling for invalid channel names."""
    with pytest.raises(ValueError, match="Channel 'invalid-channel' not found"):
        await discord_tools.read_channel(
            channel_name="invalid-channel",
            limit=5
        )

@pytest.mark.asyncio
async def test_service_error(discord_tools):
    """Test handling of service errors."""
    # Setup mock response with error status
    mock_response = AsyncMock()
    mock_response.status = 500

    # Setup session mock with proper async context managers
    mock_session = AsyncMock()
    mock_session.get = AsyncMock()
    mock_session.get.return_value = mock_response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(Exception, match="Failed to get messages: 500"):
            await discord_tools.read_channel(
                channel_name="ai-news",
                limit=5
            )

def test_channel_validation(discord_tools):
    """Test channel name validation and cleaning."""
    # Verify channel name cleaning (stripping #)
    assert discord_tools.channels.get("ai-news") == TEST_CHANNELS["ai-news"]
    
    # Verify all test channels are accessible
    for channel_name in TEST_CHANNELS:
        assert channel_name in discord_tools.channels
        assert discord_tools.channels[channel_name] == TEST_CHANNELS[channel_name]

def test_categories_structure(discord_tools):
    """Test category structure matches config."""
    assert discord_tools.categories == TEST_CATEGORIES
    
    # Verify all channels in categories exist in channels map
    for category, channels in TEST_CATEGORIES.items():
        for channel in channels:
            assert channel in discord_tools.channels 