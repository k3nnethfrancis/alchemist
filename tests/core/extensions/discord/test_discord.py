"""Test suite for Discord extension components."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import discord

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
    discord_client.user = MagicMock()
    discord_client.user.name = "TestBot"
    discord_client.user.id = "789"
    
    with patch('logging.Logger.info') as mock_logger:
        await discord_client.on_ready()
        mock_logger.assert_called_once()

@pytest.mark.asyncio
async def test_client_message_processing(discord_client, mock_message):
    """Test message processing with different scenarios."""
    # Test bot message
    mock_message.author.bot = True
    await discord_client.on_message(mock_message)
    discord_client.agent._step.assert_not_called()
    
    # Test normal message without mention
    mock_message.author.bot = False
    discord_client.user = MagicMock()
    discord_client.user.mentioned_in.return_value = False
    await discord_client.on_message(mock_message)
    discord_client.agent._step.assert_not_called()
    
    # Test mention message
    discord_client.user.mentioned_in.return_value = True
    discord_client.user.id = "789"
    mock_message.content = "<@789> hello"
    discord_client.agent._step.return_value = "Test response"
    await discord_client.on_message(mock_message)
    discord_client.agent._step.assert_called_once()

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