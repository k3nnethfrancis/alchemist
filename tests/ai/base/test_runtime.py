"""Test suite for Runtime system."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import BaseModel

from alchemist.ai.base.agent import BaseAgent

from alchemist.ai.base.runtime import (
    RuntimeConfig, 
    BaseRuntime, 
    BaseChatRuntime,
    LocalRuntime
)
from alchemist.ai.prompts.persona import AUG_E

@pytest.fixture
def runtime_config():
    """Create a runtime configuration for testing."""
    return RuntimeConfig(
        provider="openpipe",
        persona=AUG_E,
        tools=[],
        platform_config={"test": "config"}
    )

@pytest.fixture
def local_runtime(runtime_config):
    """Create a local runtime for testing."""
    return LocalRuntime(runtime_config)

@pytest.mark.asyncio
async def test_runtime_initialization(local_runtime):
    """Test runtime initialization."""
    assert local_runtime.config.provider == "openpipe"
    assert local_runtime.agent is not None
    assert local_runtime.current_session is None

@pytest.mark.asyncio
async def test_session_management(local_runtime):
    """Test session management."""
    local_runtime._start_session("test")
    assert local_runtime.current_session is not None
    assert local_runtime.current_session.platform == "test"

@pytest.mark.asyncio
async def test_message_processing(local_runtime):
    """Test message processing."""
    with patch('alchemist.ai.base.agent.BaseAgent._step') as mock_step:
        mock_step.return_value = "Test response"
        response = await local_runtime.process_message("Test message")
        assert response == "Test response"
        mock_step.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_local_runtime_formatting(local_runtime):
    """Test message formatting in local runtime."""
    test_message = "Line 1\nLine 2\n![Image](url)"
    formatted = local_runtime._format_message(
        test_message, 
        prefix="Test: "
    )
    assert "Test: Line 1" in formatted
    assert "🖼️ Generated Image" in formatted

@pytest.mark.asyncio
async def test_runtime_error_handling(local_runtime):
    """Test runtime error handling."""
    with patch('alchemist.ai.base.agent.BaseAgent._step', 
              side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            await local_runtime.process_message("Test")

@pytest.mark.asyncio
async def test_runtime_session_lifecycle(base_runtime_config):
    """Test complete session lifecycle."""
    runtime = LocalRuntime(base_runtime_config)
    
    # Mock input/output
    with patch('builtins.input', return_value="quit"), \
         patch('builtins.print') as mock_print:
        
        # Start runtime
        await runtime.start()
        assert runtime.current_session is not None
        assert runtime.current_session.platform == "local"
        
        # Verify welcome message
        welcome_calls = [
            call for call in mock_print.mock_calls 
            if isinstance(call.args[0], str) and "Starting chat" in call.args[0]
        ]
        assert welcome_calls
        
        # Verify session ended
        end_calls = [
            call for call in mock_print.mock_calls 
            if isinstance(call.args[0], str) and "session ended" in call.args[0]
        ]
        assert end_calls

@pytest.mark.asyncio
async def test_runtime_message_formatting(base_runtime_config):
    """Test message formatting with different content types."""
    runtime = LocalRuntime(base_runtime_config)
    test_cases = [
        ("Simple text", "Simple text"),
        ("Multi\nline\ntext", "Multi\nline\ntext"),
        ("![Image](url)", "🖼️ Generated Image"),
        ("Text with ![Image](url) embedded", "Text with"),
        ("```code\nblock\n```", "```code\nblock\n```")
    ]
    
    for input_text, expected_content in test_cases:
        formatted = runtime._format_message(input_text)
        assert expected_content in formatted

@pytest.mark.asyncio
async def test_runtime_error_handling_scenarios(base_runtime_config):
    """Test various error handling scenarios."""
    runtime = LocalRuntime(base_runtime_config)
    runtime._start_session("local")
    
    # Test regular errors first
    error_cases = [
        (ValueError("Invalid input"), "Error"),
        (Exception("Unknown error"), "Error")
    ]
    
    for error, expected_message in error_cases:
        runtime.agent._step = AsyncMock(side_effect=error)
        with pytest.raises(Exception) as exc_info:
            await runtime.process_message("Test")
        assert str(error) in str(exc_info.value)

    # Test KeyboardInterrupt separately
    runtime.agent._step = AsyncMock(side_effect=KeyboardInterrupt())
    with patch('builtins.print') as mock_print:
        with pytest.raises(KeyboardInterrupt):
            await runtime.process_message("Test")

@pytest.mark.asyncio
async def test_runtime_tool_integration(base_runtime_config):
    """Test runtime handling of tool responses."""
    runtime = LocalRuntime(base_runtime_config)
    runtime._start_session("local")
    
    mock_tool_response = "Tool execution result"
    
    with patch.object(BaseAgent, '_step') as mock_step:
        mock_step.return_value = f"Using tool: {mock_tool_response}"
        response = await runtime.process_message("Use tool")
        formatted = runtime._format_message(response)
        assert mock_tool_response in formatted

@pytest.mark.asyncio
async def test_runtime_config_validation():
    """Test runtime configuration validation."""
    invalid_configs = [
        {"provider": "invalid"},
        {"persona": None},
        {"tools": "not_a_list"}
    ]
    
    for invalid_config in invalid_configs:
        with pytest.raises(Exception):
            RuntimeConfig(**invalid_config)