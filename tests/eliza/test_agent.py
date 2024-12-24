"""
Test suite for the ElizaAgent class.

Tests the core functionality of the Eliza agent including:
- Decision making
- Message processing
- Cooldown behavior
- Behavior loop
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from ai.agents.eliza.agent import ElizaAgent, ElizaConfig

@pytest.fixture
def eliza_agent():
    """Create an ElizaAgent instance for testing."""
    agent = ElizaAgent(
        provider="openpipe",
        persona=[
            "Responds with short, punchy messages",
            "Uses emojis sparingly",
            "Maintains a friendly but mysterious tone"
        ],
        cooldown_seconds=3,
        scan_interval_seconds=120
    )
    return agent

@pytest.fixture
def sample_messages():
    """Fixture providing sample messages for testing."""
    return [
        {
            "id": "1",
            "author": "user1",
            "content": "Hey everyone!",
            "timestamp": datetime.now().isoformat(),
            "mentions": [],
            "referenced_message": None
        },
        {
            "id": "2",
            "author": "user2",
            "content": "What's up?",
            "timestamp": datetime.now().isoformat(),
            "mentions": [],
            "referenced_message": None
        }
    ]

@pytest.fixture
def sample_channel_context():
    """Fixture providing sample channel context for testing."""
    return "Channel: #general\\nTopic: General discussion\\nCategory: Main"

@pytest.mark.asyncio
async def test_eliza_agent_initialization(eliza_agent):
    """Test that ElizaAgent initializes correctly."""
    assert isinstance(eliza_agent.config, ElizaConfig)
    assert eliza_agent.config.provider == "openpipe"
    assert len(eliza_agent.config.persona) == 3
    assert isinstance(eliza_agent.cooldown, timedelta)
    assert isinstance(eliza_agent.scan_interval, timedelta)
    assert eliza_agent.cooldown.total_seconds() == 3
    assert eliza_agent.scan_interval.total_seconds() == 120
    assert isinstance(eliza_agent.history, list)

@pytest.mark.asyncio
async def test_should_respond_positive(eliza_agent, sample_messages, sample_channel_context):
    """Test should_respond when it should engage."""
    mock_response = {
        "should_respond": True,
        "reasoning": "Natural conversation opening",
        "response_type": "direct"
    }
    
    with patch('ai.agents.eliza.agent.ElizaAgent._call_provider', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_response
        
        with patch.object(ElizaAgent, '_step', new_callable=AsyncMock) as mock_step:
            mock_step.return_value = "Hello!"
            response = await eliza_agent.process_new_messages(sample_messages, sample_channel_context)
            assert response is not None
            mock_step.assert_called_once_with(sample_messages[-1]["content"])

@pytest.mark.asyncio
async def test_should_respond_negative(eliza_agent, sample_messages, sample_channel_context):
    """Test should_respond when it should not engage."""
    mock_response = {
        "should_respond": False,
        "reasoning": "No natural opening",
        "response_type": None
    }
    
    with patch('ai.agents.eliza.agent.ElizaAgent._call_provider', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_response
        response = await eliza_agent.process_new_messages(sample_messages, sample_channel_context)
        assert response is None

@pytest.mark.asyncio
async def test_cooldown_behavior(eliza_agent, sample_messages, sample_channel_context):
    """Test that cooldown prevents rapid responses."""
    mock_response = {
        "should_respond": True,
        "reasoning": "Natural conversation opening",
        "response_type": "direct"
    }
    
    with patch('ai.agents.eliza.agent.ElizaAgent._call_provider', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_response
        
        with patch.object(ElizaAgent, '_step', new_callable=AsyncMock) as mock_step:
            mock_step.return_value = "Hello!"
            
            # First call should work
            response1 = await eliza_agent.process_new_messages(sample_messages, sample_channel_context)
            assert response1 is not None
            
            # Immediate second call should be blocked by cooldown
            response2 = await eliza_agent.process_new_messages(sample_messages, sample_channel_context)
            assert response2 is None
            
            # Wait for cooldown
            await asyncio.sleep(3)
            
            # Third call should work
            response3 = await eliza_agent.process_new_messages(sample_messages, sample_channel_context)
            assert response3 is not None

@pytest.mark.asyncio
async def test_behavior_loop(eliza_agent):
    """Test the behavior loop's basic functionality."""
    # Mock the sleep to avoid waiting
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        # Set up a way to break the loop after one iteration
        mock_sleep.side_effect = KeyboardInterrupt()
        
        try:
            await eliza_agent.run_behavior_loop()
        except KeyboardInterrupt:
            pass
        
        mock_sleep.assert_called_once_with(120)  # Should try to sleep for scan_interval

@pytest.mark.asyncio
async def test_process_new_messages_error_handling(eliza_agent, sample_messages, sample_channel_context):
    """Test error handling in process_new_messages."""
    with patch.object(ElizaAgent, 'should_respond', side_effect=Exception("Test error")):
        response = await eliza_agent.process_new_messages(sample_messages, sample_channel_context)
        assert response is None  # Should handle error gracefully

@pytest.mark.asyncio
async def test_response_generation(eliza_agent, sample_messages, sample_channel_context):
    """Test that responses are generated correctly when should_respond is True."""
    mock_decision = {
        "should_respond": True,
        "reasoning": "Good opportunity to engage",
        "response_type": "direct"
    }
    
    expected_response = "Hello! How can I help!"
    
    with patch('ai.agents.eliza.agent.ElizaAgent._call_provider', new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            mock_decision,  # First call for should_respond
            {"content": expected_response}  # Second call for _step
        ]
        
        response = await eliza_agent.process_new_messages(sample_messages, sample_channel_context)
        
        assert response == expected_response
        assert len(mock_call.mock_calls) == 2

@pytest.mark.asyncio
async def test_history_management(eliza_agent):
    """Test that message history is properly maintained."""
    message = "Hello, world!"
    expected_response = "Hi there!"
    
    with patch('ai.agents.eliza.agent.ElizaAgent._call_provider', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"content": expected_response}
        
        response = await eliza_agent._step(message)
        
        assert response == expected_response
        assert len(eliza_agent.history) == 2  # User message and agent response
        assert eliza_agent.history[0]["role"] == "user"
        assert eliza_agent.history[0]["content"] == message
        assert eliza_agent.history[1]["role"] == "assistant"
        assert eliza_agent.history[1]["content"] == expected_response 