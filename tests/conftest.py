"""Configure pytest for the alchemist package tests."""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, AsyncMock
import discord
from dotenv import load_dotenv

from alchemist.ai.base.runtime import RuntimeConfig
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.persona import AUG_E

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables for tests
load_dotenv()

@pytest.fixture
def mock_discord_message():
    """Create a mock Discord message for testing."""
    message = MagicMock(spec=discord.Message)
    message.content = "Test message"
    message.author = MagicMock(
        bot=False,
        id="123",
        name="TestUser"
    )
    message.channel = MagicMock(
        id="456",
        name="test-channel",
        send=AsyncMock()
    )
    message.mentions = []
    message.created_at = MagicMock()
    return message

@pytest.fixture
def base_runtime_config():
    """Create a base runtime configuration for testing."""
    return RuntimeConfig(
        provider="openpipe",
        persona=AUG_E,
        tools=[],
        platform_config={}
    )

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent._step = AsyncMock(return_value="Test response")
    agent.provider = "openpipe"
    return agent

@pytest.fixture
def discord_intents():
    """Create Discord intents for testing."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    return intents 