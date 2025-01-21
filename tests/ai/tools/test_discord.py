"""Tests for the Discord tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from mirascope.core import BaseMessageParam
from alchemist.ai.tools.discord_tool import DiscordReaderTool

def test_tool_metadata():
    """Test tool name and description."""
    tool = DiscordReaderTool(query="test", channel="general")
    assert tool._name() == "read_discord"
    assert "discord" in tool._description().lower()
    assert "channel" in tool._description().lower()

def test_validation():
    """Test configuration validation."""
    # Should raise error if neither token nor service_url provided
    with pytest.raises(ValueError, match="Either token.*or service_url.*must be provided"):
        tool = DiscordReaderTool(query="test", channel="general")
        tool._validate_config()
    
    # Should not raise with token
    tool = DiscordReaderTool(query="test", channel="general", token="test-token")
    tool._validate_config()
    
    # Should not raise with service_url
    tool = DiscordReaderTool(query="test", channel="general", service_url="http://test")
    tool._validate_config()

@pytest.mark.asyncio
async def test_service_mode():
    """Test service mode operation."""
    with patch("alchemist.ai.tools.discord_tool.aiohttp.ClientSession") as mock_session:
        # Mock session
        session = AsyncMock()
        mock_session.return_value = session
        
        # Mock responses
        channels_response = AsyncMock()
        channels_response.status = 200
        channels_response.json = AsyncMock(return_value={
            "channels": {"test-channel": "123456"}
        })
        
        messages_response = AsyncMock()
        messages_response.status = 200
        messages_response.json = AsyncMock(return_value={
            "messages": ["message1", "message2"]
        })
        
        # Set up get method to return responses
        session.get = AsyncMock()
        session.get.side_effect = [channels_response, messages_response]
        
        tool = DiscordReaderTool(
            query="test query",
            channel="test-channel",
            service_url="http://test"
        )
        
        result = await tool.call()
        assert result == ["message1", "message2"]

@pytest.mark.asyncio
async def test_direct_mode():
    """Test direct Discord API mode."""
    with patch("alchemist.ai.tools.discord_tool.commands.Bot") as mock_bot_class:
        # Mock bot instance
        mock_bot = AsyncMock()
        mock_bot_class.return_value = mock_bot
        
        # Mock channel
        mock_channel = AsyncMock()
        mock_message = MagicMock()
        mock_message.content = "test message"
        mock_message.author.name = "user"
        mock_message.created_at = datetime.utcnow()
        
        # Set up async iterator for history
        mock_channel.history = AsyncMock()
        mock_channel.history.return_value = AsyncMock()
        mock_channel.history.return_value.__aiter__ = AsyncMock(return_value=AsyncMock())
        mock_channel.history.return_value.__aiter__.return_value.__anext__ = AsyncMock(
            side_effect=[mock_message, StopAsyncIteration]
        )
        
        # Set up bot methods
        mock_bot.get_channel.return_value = mock_channel
        mock_bot.login = AsyncMock()
        mock_bot.connect = AsyncMock()
        
        tool = DiscordReaderTool(
            query="test",
            channel="123456",
            token="test-token"
        )
        
        result = await tool.call()
        assert len(result) == 1
        assert "test message" in result[0]
        assert "user" in result[0] 