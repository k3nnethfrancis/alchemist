"""
Test suite for the ChatRuntime class.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from ai.base.runtime import BaseRuntime
from ai.agents.chat.runtime import ChatRuntime
from ai.agents.chat.agent import ChatAgent
from ai.prompts.persona import AUG_E

@pytest.fixture
def chat_runtime():
    """Fixture to create a ChatRuntime instance for testing."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    runtime = ChatRuntime(agent=agent)
    runtime._start_session("cli")  # Ensure session is started
    return runtime

def test_base_runtime_abstract():
    """Test that BaseRuntime cannot be instantiated directly."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseRuntime"):
        BaseRuntime(agent=agent)

@pytest.mark.asyncio
async def test_local_chat_example():
    """Test the local chat example end-to-end."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    runtime = ChatRuntime(agent=agent)
    
    # Mock input/output
    with patch('builtins.input', side_effect=["Hello", "quit"]), \
         patch('builtins.print') as mock_print, \
         patch.object(ChatAgent, '_step', new_callable=AsyncMock) as mock_step:
        
        mock_step.return_value = "Test response"
        
        # Run start/stop sequence
        await runtime.start()
        await runtime.stop()
        
        # Verify session messages
        mock_print.assert_any_call("🌟 Chat Session Started 🌟")
        mock_print.assert_any_call("🌟 Chat Session Ended 🌟")
        
        # Verify agent interaction
        mock_step.assert_called_once_with("Hello")

@pytest.mark.asyncio
async def test_runtime_initialization(chat_runtime):
    """Test that ChatRuntime initializes correctly."""
    assert chat_runtime.agent is not None
    assert isinstance(chat_runtime.agent, ChatAgent)
    assert chat_runtime.config == {}  # Default empty config
    assert chat_runtime.current_session is not None
    assert chat_runtime.current_session.platform == "cli"  # Default platform

@pytest.mark.asyncio
async def test_runtime_custom_config():
    """Test ChatRuntime with custom configuration."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    config = {"test_key": "test_value"}
    runtime = ChatRuntime(agent=agent, config=config)
    assert runtime.config == config

@pytest.mark.asyncio
async def test_runtime_process_message():
    """Test the ChatRuntime's message processing."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    runtime = ChatRuntime(agent=agent)
    runtime._start_session("cli")
    
    # Mock the agent's step method
    with patch.object(ChatAgent, '_step', new_callable=AsyncMock) as mock_step:
        mock_step.return_value = "Test response"
        response = await runtime.process_message("Hello")
        assert response == "Test response"
        mock_step.assert_called_once_with("Hello")

@pytest.mark.asyncio
async def test_runtime_start():
    """Test the ChatRuntime start method."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    runtime = ChatRuntime(agent=agent)
    
    # Mock input/output
    with patch('builtins.input', side_effect=["Hello", "quit"]), \
         patch('builtins.print') as mock_print, \
         patch.object(ChatAgent, '_step', new_callable=AsyncMock) as mock_step:
        
        mock_step.return_value = "Test response"
        await runtime.start()
        
        # Verify the session started message was printed
        mock_print.assert_any_call("\n" + "="*50)
        mock_print.assert_any_call("🌟 Chat Session Started 🌟")

@pytest.mark.asyncio
async def test_runtime_stop():
    """Test the ChatRuntime stop method."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    runtime = ChatRuntime(agent=agent)
    
    # Mock print function
    with patch('builtins.print') as mock_print:
        await runtime.stop()
        
        # Verify the session ended message was printed
        mock_print.assert_any_call("\n" + "="*50)
        mock_print.assert_any_call("🌟 Chat Session Ended 🌟")

@pytest.mark.asyncio
async def test_runtime_error_handling():
    """Test ChatRuntime's error handling."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    runtime = ChatRuntime(agent=agent)
    
    # Mock input/output and force an error
    with patch('builtins.input', side_effect=["Hello", "quit"]), \
         patch('builtins.print') as mock_print, \
         patch.object(ChatAgent, '_step', side_effect=Exception("Test error")):
        
        await runtime.start()
        
        # Verify error was printed - check for error message in any format
        error_calls = [
            call for call in mock_print.mock_calls 
            if isinstance(call.args[0], str) and "Test error" in call.args[0]
        ]
        assert error_calls, "No error message was printed"

@pytest.mark.asyncio
async def test_runtime_platform_change():
    """Test changing the runtime platform."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    runtime = ChatRuntime(agent=agent)
    
    # Change platform
    runtime._start_session("discord")
    assert runtime.current_session is not None
    assert runtime.current_session.platform == "discord"
    
    # Process a message with the new platform
    with patch.object(ChatAgent, '_step', new_callable=AsyncMock) as mock_step:
        mock_step.return_value = "Test response"
        response = await runtime.process_message("Hello")
        assert response == "Test response" 