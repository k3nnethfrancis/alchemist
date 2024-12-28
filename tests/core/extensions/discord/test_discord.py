"""Test suite for Discord extension components."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock
import discord
from datetime import datetime

from alchemist.core.extensions.config import get_discord_config, get_extension_config
from alchemist.core.extensions.discord.client import DiscordClient
from alchemist.core.extensions.discord.runtime import DiscordRuntime
from alchemist.ai.base.runtime import RuntimeConfig
from alchemist.ai.prompts.persona import AUG_E

# Config Tests
def test_get_discord_config():
    """Test Discord configuration retrieval."""
    with patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test-token'}):
        config = get_discord_config()
        assert config['token'] == 'test-token'
        assert isinstance(config['intents'], discord.Intents)
        assert config['intents'].message_content is True

def test_get_discord_config_missing_token():
    """Test Discord configuration with missing token."""
    with patch.dict('os.environ', clear=True):
        with pytest.raises(ValueError) as exc:
            get_discord_config()
        assert "DISCORD_BOT_TOKEN" in str(exc.value)

def test_get_extension_config():
    """Test extension configuration retrieval."""
    with patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test-token'}):
        config = get_extension_config('discord')
        assert 'token' in config
        assert 'intents' in config

def test_get_extension_config_invalid():
    """Test invalid extension configuration retrieval."""
    with pytest.raises(ValueError) as exc:
        get_extension_config('invalid')
    assert "Unsupported extension" in str(exc.value)

# Discord Client Tests
@pytest.fixture
def discord_client():
    """Create a Discord client for testing."""
    agent = MagicMock()
    intents = discord.Intents.default()
    return DiscordClient(agent=agent, intents=intents, token="test-token")

@pytest.fixture
def mock_message():
    """Create a mock Discord message."""
    message = MagicMock(spec=discord.Message)
    message.content = "Test message"
    message.author.bot = False
    message.author.id = "123"
    message.author.name = "TestUser"
    message.channel.id = "456"
    message.channel.name = "test-channel"
    message.mentions = []
    message.created_at = MagicMock()
    return message

@pytest.mark.asyncio
async def test_client_setup_hook(discord_client):
    """Test client setup hook."""
    with patch('logging.Logger.info') as mock_logger:
        await discord_client.setup_hook()
        mock_logger.assert_called_once()

@pytest.mark.asyncio
async def test_client_on_ready(discord_client):
    """Test client ready event."""
    # Create a mock user
    mock_user = MagicMock(spec=discord.User)
    mock_user.name = "TestBot"
    mock_user.id = "123456789"
    
    # Mock the user property
    type(discord_client).user = PropertyMock(return_value=mock_user)
    
    with patch('logging.Logger.info') as mock_logger:
        await discord_client.on_ready()
        mock_logger.assert_called_once_with(f"Logged in as TestBot (123456789)")

@pytest.mark.asyncio
async def test_client_message_processing(discord_client, mock_message):
    """Test message processing with different scenarios."""
    # Create a mock user
    mock_user = MagicMock(spec=discord.User)
    mock_user.id = "123456789"
    mock_user.mentioned_in = MagicMock(return_value=False)
    
    # Mock the user property
    type(discord_client).user = PropertyMock(return_value=mock_user)
    
    # Set up mock message attributes
    mock_message.channel.send = AsyncMock()
    mock_message.created_at = datetime.now()
    mock_message.content = "test message"
    mock_message.mentions = []
    mock_message.author.id = "123"
    mock_message.author.name = "TestUser"
    mock_message.channel.id = "456"
    mock_message.channel.name = "test-channel"
    
    # Remove process_discord_message to test standard flow
    delattr(discord_client.agent, 'process_discord_message')
    discord_client.agent._step = AsyncMock(return_value="Test response")
    
    # Test bot message
    mock_message.author.bot = True
    await discord_client.on_message(mock_message)
    discord_client.agent._step.assert_not_called()
    
    # Test normal message without mention
    mock_message.author.bot = False
    mock_user.mentioned_in.return_value = False
    await discord_client.on_message(mock_message)
    discord_client.agent._step.assert_not_called()
    
    # Test mention message
    mock_user.mentioned_in.return_value = True
    mock_message.content = "<@123456789> hello"
    await discord_client.on_message(mock_message)
    discord_client.agent._step.assert_called_once_with("hello")
    mock_message.channel.send.assert_awaited_once_with("Test response")

# Discord Runtime Tests
@pytest.fixture
def runtime_config():
    """Create runtime configuration for testing."""
    return RuntimeConfig(
        provider="openpipe",
        persona=AUG_E,
        tools=[],
        platform_config={
            "intents": ["message_content"],
            "activity_type": "listening",
            "activity_name": "mentions"
        }
    )

@pytest.mark.asyncio
async def test_discord_runtime_initialization(runtime_config):
    """Test Discord runtime initialization."""
    runtime = DiscordRuntime(token="test-token", config=runtime_config)
    assert runtime.token == "test-token"
    assert runtime.client is None
    assert runtime.agent is not None

@pytest.mark.asyncio
async def test_discord_runtime_start(runtime_config):
    """Test Discord runtime start."""
    runtime = DiscordRuntime(token="test-token", config=runtime_config)
    
    with patch('discord.Client.start') as mock_start:
        await runtime.start()
        assert runtime.current_session is not None
        assert runtime.current_session.platform == "discord"
        assert runtime.client is not None
        mock_start.assert_called_once_with("test-token")

@pytest.mark.asyncio
async def test_discord_runtime_stop(runtime_config):
    """Test Discord runtime stop."""
    runtime = DiscordRuntime(token="test-token", config=runtime_config)
    runtime.client = MagicMock()
    runtime.client.close = AsyncMock()
    
    await runtime.stop()
    runtime.client.close.assert_called_once()

@pytest.mark.asyncio
async def test_discord_runtime_agent_creation(runtime_config):
    """Test agent creation in Discord runtime."""
    runtime = DiscordRuntime(token="test-token", config=runtime_config)
    agent = runtime._create_agent()
    assert agent.provider == runtime_config.provider
    assert agent.history[0].role == "system"  # Check system prompt