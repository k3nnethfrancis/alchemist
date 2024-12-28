"""Test suite for BaseAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from mirascope.core import BaseMessageParam

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.persona import AUG_E
from alchemist.ai.base.tools import ImageGenerationTool

@pytest.fixture
def base_agent():
    """Create a base agent for testing."""
    return BaseAgent(provider="openpipe", persona=AUG_E)

@pytest.mark.asyncio
async def test_agent_initialization(base_agent):
    """Test agent initialization and system prompt."""
    assert base_agent.provider == "openpipe"
    assert len(base_agent.history) == 1  # System prompt
    assert base_agent.history[0].role == "system"

@pytest.mark.asyncio
async def test_agent_step():
    """Test agent step processing."""
    agent = BaseAgent(provider="openpipe")
    
    with patch('alchemist.ai.base.agent.BaseAgent._call_openpipe') as mock_call:
        mock_call.return_value = AsyncMock(
            content="Test response",
            message_param=BaseMessageParam(role="assistant", content="Test response"),
            tool=None
        )
        
        response = await agent._step("Hello")
        assert response == "Test response"
        assert len(agent.history) == 3  # System + user + assistant

@pytest.mark.asyncio
async def test_agent_tool_execution():
    """Test agent tool execution."""
    agent = BaseAgent(provider="openpipe")
    mock_tool = AsyncMock(
        _name=lambda: "test_tool",
        args={"arg": "value"},
        call=AsyncMock(return_value="Tool result")
    )
    
    with patch('alchemist.ai.base.agent.BaseAgent._call_openpipe') as mock_call:
        mock_call.side_effect = [
            AsyncMock(
                content="Using tool",
                message_param=BaseMessageParam(role="assistant", content="Using tool"),
                tool=mock_tool,
                tool_message_params=lambda result: [
                    BaseMessageParam(role="tool", content=str(result[0][1]))
                ]
            ),
            AsyncMock(
                content="Tool response",
                message_param=BaseMessageParam(role="assistant", content="Tool response"),
                tool=None
            )
        ]
        
        response = await agent._step("Use tool")
        assert response == "Tool response"
        mock_tool.call.assert_called_once()

@pytest.mark.asyncio
async def test_agent_provider_selection():
    """Test provider selection and calling."""
    providers = ["openai", "anthropic", "openpipe"]
    
    for provider in providers:
        agent = BaseAgent(provider=provider)
        mock_method = f"_call_{provider}"
        
        with patch(f'alchemist.ai.base.agent.BaseAgent.{mock_method}') as mock_call:
            mock_call.return_value = AsyncMock(
                content="Test response",
                message_param=BaseMessageParam(role="assistant", content="Test response"),
                tool=None
            )
            
            response = await agent._step("Hello")
            assert response == "Test response"
            mock_call.assert_called_once()

@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test agent error handling."""
    agent = BaseAgent(provider="openpipe")
    
    with patch('alchemist.ai.base.agent.BaseAgent._call_openpipe', 
              side_effect=Exception("Test error")):
        response = await agent._step("Hello")
        assert "error" in response.lower()

@pytest.mark.asyncio
async def test_agent_message_history_management():
    """Test message history management."""
    agent = BaseAgent(provider="openpipe")
    initial_history_len = len(agent.history)
    
    with patch('alchemist.ai.base.agent.BaseAgent._call_openpipe') as mock_call:
        mock_call.return_value = AsyncMock(
            content="Test response",
            message_param=BaseMessageParam(role="assistant", content="Test response"),
            tool=None
        )
        
        # Test message addition
        await agent._step("Hello")
        assert len(agent.history) == initial_history_len + 2  # User + Assistant
        assert agent.history[-2].role == "user"
        assert agent.history[-1].role == "assistant"
        
        # Test message content
        assert agent.history[-2].content == "Hello"
        assert agent.history[-1].content == "Test response"

@pytest.mark.asyncio
async def test_agent_tool_chain():
    """Test tool chaining behavior."""
    agent = BaseAgent(provider="openpipe")
    mock_tool1 = AsyncMock(
        _name=lambda: "tool1",
        args={"arg": "value1"},
        call=AsyncMock(return_value="Tool1 result")
    )
    mock_tool2 = AsyncMock(
        _name=lambda: "tool2",
        args={"arg": "value2"},
        call=AsyncMock(return_value="Tool2 result")
    )
    
    with patch('alchemist.ai.base.agent.BaseAgent._call_openpipe') as mock_call:
        mock_call.side_effect = [
            # First tool response
            AsyncMock(
                content="Using tool1",
                message_param=BaseMessageParam(role="assistant", content="Using tool1"),
                tool=mock_tool1,
                tool_message_params=lambda result: [
                    BaseMessageParam(role="tool", content=str(result[0][1]))
                ]
            ),
            # Second tool response
            AsyncMock(
                content="Using tool2",
                message_param=BaseMessageParam(role="assistant", content="Using tool2"),
                tool=mock_tool2,
                tool_message_params=lambda result: [
                    BaseMessageParam(role="tool", content=str(result[0][1]))
                ]
            ),
            # Final response
            AsyncMock(
                content="Final response",
                message_param=BaseMessageParam(role="assistant", content="Final response"),
                tool=None
            )
        ]
        
        response = await agent._step("Use tools")
        assert response == "Final response"
        mock_tool1.call.assert_called_once()
        mock_tool2.call.assert_called_once()

@pytest.mark.asyncio
async def test_agent_provider_specific_behavior():
    """Test provider-specific behaviors."""
    providers = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "openpipe": "gpt-4o-mini"
    }
    
    for provider, model in providers.items():
        agent = BaseAgent(provider=provider)
        mock_method = f"_call_{provider}"
        
        with patch(f'alchemist.ai.base.agent.BaseAgent.{mock_method}') as mock_call:
            await agent._step("Test")
            # Verify correct model and configuration
            call_args = mock_call.call_args[1] if mock_call.call_args[1] else mock_call.call_args[0]
            assert len(call_args) > 0  # Ensure call was made with arguments

@pytest.mark.asyncio
async def test_image_generation_tool():
    """Test image generation tool integration."""
    agent = BaseAgent(provider="openpipe")
    mock_image_tool = AsyncMock(
        _name=lambda: "image_generation",
        args={"prompt": "test image"},
        call=AsyncMock(return_value="http://test.image.url")
    )
    
    with patch('alchemist.ai.base.agent.BaseAgent._call_openpipe') as mock_call:
        # Set up the mock to return a complete response sequence
        mock_call.side_effect = [
            AsyncMock(
                content="Generating image",
                message_param=BaseMessageParam(role="assistant", content="Generating image"),
                tool=mock_image_tool,
                tool_message_params=lambda result: [
                    BaseMessageParam(role="tool", content="http://test.image.url")
                ]
            ),
            # Final response after tool execution
            AsyncMock(
                content="Here's your generated image: http://test.image.url",
                message_param=BaseMessageParam(
                    role="assistant", 
                    content="Here's your generated image: http://test.image.url"
                ),
                tool=None
            )
        ]
        
        response = await agent._step("Generate an image")
        assert mock_image_tool.call.called
        assert "http://test.image.url" in response

@pytest.mark.asyncio
async def test_agent_error_recovery():
    """Test agent error recovery and graceful degradation."""
    agent = BaseAgent(provider="openpipe")
    
    with patch('alchemist.ai.base.agent.BaseAgent._call_openpipe') as mock_call:
        # Simulate different types of errors
        errors = [
            ValueError("Invalid input"),
            ConnectionError("Network error"),
            Exception("Unknown error")
        ]
        
        for error in errors:
            mock_call.side_effect = error
            response = await agent._step("Test")
            assert "error" in response.lower()
            assert str(error) in response