"""Test suite for BaseAgent implementation.

This module contains tests for the BaseAgent class, verifying:
1. Agent initialization and configuration
2. Message handling and history management
3. Tool integration and execution
4. Provider-specific call handling
"""

import pytest
from unittest.mock import AsyncMock, patch
from mirascope.core import BaseMessageParam

from alchemist.ai.base.agent import BaseAgent, CalculatorTool
from alchemist.ai.prompts.persona import AUG_E
from alchemist.ai.prompts.base import create_system_prompt

@pytest.fixture
def mock_call_response():
    """Create a mock call response for testing."""
    return AsyncMock(
        content="Test response",
        message_param=BaseMessageParam(role="assistant", content="Test response"),
        tool=None,
        tool_message_params=None,
        tools=None
    )

@pytest.fixture
def mock_tool_response():
    """Create a mock tool call response for testing."""
    tool = CalculatorTool(expression="2 + 2")
    return AsyncMock(
        content="Let me calculate that for you",
        message_param=BaseMessageParam(
            role="assistant",
            content="Let me calculate that for you",
            function_call={"name": "calculator", "arguments": '{"expression": "2 + 2"}'}
        ),
        tool=tool,
        tool_message_params=lambda result: [
            BaseMessageParam(
                role="function",
                name="calculator",
                content=str(result[0][1])
            )
        ],
        tools=None
    )

@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization with proper configuration."""
    agent = BaseAgent(
        provider="openpipe",
        model="gpt-4o-mini",
        persona=AUG_E,
        tools=[CalculatorTool]
    )
    
    # Verify basic configuration
    assert agent.provider == "openpipe"
    assert agent.model == "gpt-4o-mini"
    assert agent.tool_classes == [CalculatorTool]
    
    # Verify system prompt is created
    system_content = create_system_prompt(agent.persona)
    assert isinstance(system_content, str)
    assert len(system_content) > 0

@pytest.mark.asyncio
async def test_prompt_creation():
    """Test prompt creation with proper message formatting."""
    agent = BaseAgent(provider="openpipe", tools=[CalculatorTool])
    
    # Add some history
    agent.history = [
        BaseMessageParam(role="user", content="Hello"),
        BaseMessageParam(role="assistant", content="Hi there")
    ]
    
    # Test prompt creation
    messages = agent._prompt("test query")
    
    # Verify message structure
    assert isinstance(messages, list)
    assert all(isinstance(msg, BaseMessageParam) for msg in messages)
    assert messages[0].role == "system"  # First message should be system
    assert len(messages) == len(agent.history) + 1  # History + system message

@pytest.mark.asyncio
async def test_basic_conversation(mock_call_response):
    """Test basic conversation flow without tools."""
    agent = BaseAgent(provider="openpipe", tools=[])
    
    with patch('mirascope.core.openai.call', return_value=lambda _: lambda _: mock_call_response):
        response = await agent._step("Hello")
        
        # Verify response
        assert response == "Test response"
        assert len(agent.history) == 2  # User message + assistant response
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"

@pytest.mark.asyncio
async def test_calculator_tool_execution(mock_tool_response, mock_call_response):
    """Test calculator tool execution and response handling."""
    agent = BaseAgent(provider="openpipe", tools=[CalculatorTool])
    
    with patch('mirascope.core.openai.call') as mock_call:
        # First call - LLM decides to use calculator
        mock_call.side_effect = [
            lambda _: lambda _: mock_tool_response,  # Returns tool call
            lambda _: lambda _: mock_call_response   # Returns final response
        ]
        
        # Execute the step
        response = await agent._step("Calculate 2 + 2")
        
        # Verify history has correct sequence:
        # 1. User message
        # 2. Assistant message (deciding to use tool)
        # 3. Function message (tool result)
        # 4. Assistant message (final response)
        assert len(agent.history) == 4
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"
        assert agent.history[2].role == "function"  # Tool result
        assert agent.history[3].role == "assistant"  # Final response
        
        # Verify tool execution
        assert mock_call.call_count == 2  # Initial call + after tool execution

@pytest.mark.asyncio
async def test_provider_specific_calls():
    """Test provider-specific call configurations."""
    providers = {
        "openai": "gpt-4",
        "anthropic": "claude-3-5-sonnet-20240620",
        "openpipe": "gpt-4o-mini"
    }
    
    for provider, model in providers.items():
        agent = BaseAgent(provider=provider, model=model, tools=[CalculatorTool])
        
        # Verify call configuration
        with patch('mirascope.core.openai.call') as mock_openai, \
             patch('mirascope.core.anthropic.call') as mock_anthropic:
            
            await agent._step("test")
            
            if provider == "anthropic":
                mock_anthropic.assert_called_once()
            else:
                mock_openai.assert_called_once()

def test_calculator_tool():
    """Test calculator tool functionality."""
    # Test basic arithmetic
    calc = CalculatorTool(expression="2 + 2")
    assert calc.call() == "4"
    
    # Test more complex expression
    calc = CalculatorTool(expression="42 ** 0.5")
    assert calc.call() == "6.48074069840786"
    
    # Test error handling
    calc = CalculatorTool(expression="invalid")
    result = calc.call()
    assert "Error" in result
    assert "name 'invalid' is not defined" in result
