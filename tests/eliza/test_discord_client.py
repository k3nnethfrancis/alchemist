"""
Test suite for the ElizaDiscordClient class.

Tests the core functionality of the Discord client including:
- Message handling
- Channel context gathering
- Error handling
"""

import os
import asyncio
import pytest
import discord
from discord.ext import commands
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ai.agents.eliza.agent import ElizaAgent
from ai.core.extensions.discord.eliza_client import ElizaDiscordClient
from ai.core.prompts.persona import AUG_E

@pytest.fixture
def mock_intents():
    """Fixture providing mock Discord intents."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    intents.guilds = True
    return intents

@pytest.fixture
def mock_agent():
    """Fixture providing a mock ElizaAgent."""
    return Mock(spec=ElizaAgent)

@pytest.fixture
def event_loop():
    """Create an event loop for testing."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def discord_client(mock_agent, mock_intents, event_loop):
    """Fixture to create an ElizaDiscordClient instance for testing."""
    with patch('discord.ext.tasks.Loop.start') as mock_start:
        client = ElizaDiscordClient(
            agent=mock_agent,
            intents=mock_intents,
            token="mock_token"
        )
        client.user = Mock(id=999)  # Mock the bot's user for mention tests
        return client

@pytest.fixture
def mock_channel():
    """Fixture providing a mock Discord channel."""
    channel = Mock(spec=discord.TextChannel)
    channel.id = 123
    channel.name = "general"
    channel.topic = "General discussion"
    category = Mock()
    category.name = "Main"
    channel.category = category
    return channel

@pytest.fixture
def mock_message(mock_channel):
    """Fixture providing a mock Discord message."""
    message = Mock(spec=discord.Message)
    message.id = 456
    message.author = Mock()
    message.author.name = "test_user"
    message.content = "Hello, world!"
    message.channel = mock_channel
    message.created_at = datetime.now()
    message.mentions = []
    message.reference = None
    return message

@pytest.mark.asyncio
async def test_client_initialization(discord_client, mock_agent):
    """Test that ElizaDiscordClient initializes correctly."""
    assert discord_client.agent == mock_agent
    assert discord_client.token == "mock_token"
    assert isinstance(discord_client.channel_history, dict)
    assert isinstance(discord_client.channel_contexts, dict)

@pytest.mark.asyncio
async def test_get_channel_context(discord_client, mock_channel):
    """Test channel context generation."""
    context = discord_client.get_channel_context(mock_channel)
    
    assert "Channel: #general" in context
    assert "Topic: General discussion" in context
    assert "Category: Main" in context
    
    # Test caching
    cached_context = discord_client.get_channel_context(mock_channel)
    assert context == cached_context
    assert mock_channel.id in discord_client.channel_contexts

@pytest.mark.asyncio
async def test_format_message(discord_client, mock_message):
    """Test message formatting."""
    formatted = discord_client.format_message(mock_message)
    
    assert formatted["id"] == 456
    assert formatted["author"] == "test_user"
    assert formatted["content"] == "Hello, world!"
    assert "timestamp" in formatted
    assert isinstance(formatted["mentions"], list)
    assert formatted["referenced_message"] is None

@pytest.mark.asyncio
async def test_on_message_mention(discord_client, mock_message, mock_agent):
    """Test message handling when bot is mentioned."""
    # Setup mention
    mock_message.mentions = [Mock(id=discord_client.user.id)]
    mock_agent.process_new_messages.return_value = "Hello!"
    
    # Process message
    await discord_client.on_message(mock_message)
    
    # Verify behavior
    assert mock_message.channel.id in discord_client.channel_history
    mock_agent.process_new_messages.assert_called_once()
    mock_message.channel.send.assert_called_once_with("Hello!")

@pytest.mark.asyncio
async def test_on_message_no_mention(discord_client, mock_message, mock_agent):
    """Test message handling when bot is not mentioned."""
    mock_agent.process_new_messages.return_value = None
    
    # Process message
    await discord_client.on_message(mock_message)
    
    # Verify behavior
    assert mock_message.channel.id in discord_client.channel_history
    assert len(discord_client.channel_history[mock_message.channel.id]) > 0
    mock_agent.process_new_messages.assert_called_once()
    mock_message.channel.send.assert_not_called()

@pytest.mark.asyncio
async def test_message_history_limit(discord_client, mock_message):
    """Test that message history is limited to 50 messages."""
    channel_id = mock_message.channel.id
    
    # Add 60 messages
    for i in range(60):
        mock_message.id = i  # Make each message unique
        await discord_client.on_message(mock_message)
    
    # Verify limit
    assert len(discord_client.channel_history[channel_id]) == 50
    # Verify it's the most recent 50
    assert discord_client.channel_history[channel_id][0]["id"] == 10

@pytest.mark.asyncio
async def test_scan_channels(discord_client, mock_channel, mock_agent):
    """Test periodic channel scanning."""
    # Setup test data
    channel_id = mock_channel.id
    discord_client.channel_history[channel_id] = [
        {"id": 1, "content": "test", "author": "user"}
    ]
    discord_client.get_channel = Mock(return_value=mock_channel)
    mock_agent.process_new_messages.return_value = "Scanning response"
    
    # Run scan
    await discord_client.scan_channels()
    
    # Verify behavior
    mock_agent.process_new_messages.assert_called_once()
    mock_channel.send.assert_called_once_with("Scanning response")

@pytest.mark.asyncio
async def test_error_handling(discord_client, mock_message, mock_agent):
    """Test error handling in message processing."""
    mock_agent.process_new_messages.side_effect = Exception("Test error")
    mock_message.guild.me = Mock()
    mock_message.channel.permissions_for.return_value.send_messages = True
    
    # Process message
    await discord_client.on_message(mock_message)
    
    # Verify error handling
    mock_message.channel.send.assert_called_once()
    assert "error" in mock_message.channel.send.call_args[0][0].lower()

@pytest.mark.asyncio
async def test_start(discord_client):
    """Test client start method."""
    with patch.object(commands.Bot, 'start') as mock_start:
        await discord_client.start()
        mock_start.assert_called_once_with("mock_token") 