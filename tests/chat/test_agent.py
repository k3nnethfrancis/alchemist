"""
Test suite for the ChatAgent class.
"""

import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict

from mirascope.core import BaseMessageParam

from ai.prompts.persona import AUG_E, Persona, Personality, PersonalityTraits, Stats, StyleGuide
from ai.agents.chat.agent import ChatAgent
from ai.agents.chat.tools import ImageGenerationTool

@pytest.fixture
def chat_agent():
    """Fixture to create a ChatAgent instance for testing."""
    return ChatAgent(provider="openpipe", persona=AUG_E)

@pytest.mark.asyncio
async def test_chat_agent_initialization(chat_agent):
    """Test that ChatAgent initializes correctly."""
    assert chat_agent.provider == "openpipe"
    assert chat_agent.persona == AUG_E
    assert chat_agent.tools == [ImageGenerationTool]
    assert len(chat_agent.history) == 1  # Should have system prompt
    assert chat_agent.history[0].role == "system"

@pytest.mark.asyncio
async def test_chat_agent_providers():
    """Test ChatAgent with different providers."""
    providers = ["openai", "anthropic", "openpipe"]
    for provider in providers:
        agent = ChatAgent(provider=provider)
        assert agent.provider == provider

@pytest.mark.asyncio
async def test_provider_specific_calls():
    """Test each provider's specific call method."""
    test_message = "Hello, how are you?"
    test_response = Mock(content="I'm doing well!", tool=None)
    
    # Test OpenAI
    with patch.object(ChatAgent, '_call_openai', return_value=test_response):
        agent = ChatAgent(provider="openai")
        response = await agent._step(test_message)
        assert response == "I'm doing well!"
    
    # Test Anthropic
    with patch.object(ChatAgent, '_call_anthropic', return_value=test_response):
        agent = ChatAgent(provider="anthropic")
        response = await agent._step(test_message)
        assert response == "I'm doing well!"
    
    # Test OpenPipe
    with patch.object(ChatAgent, '_call_openpipe', return_value=test_response):
        agent = ChatAgent(provider="openpipe")
        response = await agent._step(test_message)
        assert response == "I'm doing well!"

@pytest.mark.asyncio
async def test_provider_tool_integration():
    """Test tool integration with each provider."""
    test_message = "Generate an image of a sunset"
    image_url = "http://example.com/image.jpg"
    
    # Create response objects with proper message_param and tool_message_params
    tool_instance = ImageGenerationTool(prompt="A beautiful sunset")
    tool_response = Mock(
        content="I'll generate that image",
        tool=tool_instance,
        message_param=BaseMessageParam(role="assistant", content="I'll generate that image"),
        tool_message_params=lambda x: [
            BaseMessageParam(role="assistant", content="Tool called"),
            BaseMessageParam(role="tool", content=image_url)
        ]
    )
    followup_response = Mock(
        content="Here's your sunset",
        tool=None,
        message_param=BaseMessageParam(role="assistant", content="Here's your sunset")
    )
    
    for provider in ["openai", "anthropic", "openpipe"]:
        # Create mocks
        mock_provider = Mock(side_effect=[tool_response, followup_response])
        mock_tool = AsyncMock(return_value=image_url)
        
        # Set up patches
        with patch.object(ImageGenerationTool, 'call', new=mock_tool), \
             patch.object(ChatAgent, f'_call_{provider}', new=mock_provider):
            
            agent = ChatAgent(provider=provider)
            response = await agent._step(test_message)
            
            # Verify tool was called
            mock_tool.assert_called_once()
            assert image_url in response
            assert "![generated image]" in response

@pytest.mark.asyncio
async def test_chat_agent_custom_persona():
    """Test ChatAgent with custom persona."""
    custom_persona = Persona(
        id="test_bot",
        name="Test Bot",
        bio="A test bot",
        personality=Personality(
            traits=PersonalityTraits(
                neuroticism=0.3,
                extraversion=0.7,
                openness=0.8,
                agreeableness=0.9,
                conscientiousness=0.8
            ),
            stats=Stats(
                intelligence=0.8,
                wisdom=0.7,
                charisma=0.6,
                strength=0.5,
                dexterity=0.5,
                constitution=0.5
            )
        ),
        lore=["Created for testing", "Loves unit tests"],
        style=StyleGuide(
            all=["Be professional"],
            chat=["Be helpful", "Be concise"]
        )
    )
    agent = ChatAgent(provider="openpipe", persona=custom_persona)
    assert agent.persona == custom_persona

@pytest.mark.asyncio
async def test_chat_agent_step():
    """Test the ChatAgent's step method."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    message = "Hello, how are you?"
    response = await agent._step(message)
    assert isinstance(response, str)
    assert len(response) > 0
    assert len(agent.history) == 3  # system + user + assistant

@pytest.mark.asyncio
async def test_chat_agent_image_generation():
    """Test the ChatAgent's image generation capability."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    message = "Generate an image of a sunset"
    
    # Create mock tool instance with format_response method
    tool_instance = Mock(spec=ImageGenerationTool)
    tool_instance.format_response.return_value = "Here's your sunset image!\\n\\n![generated image](http://example.com/image.jpg)"
    
    # Create mock response
    mock_response = Mock(
        content="I'll generate that image",
        tool=tool_instance,
        message_param=BaseMessageParam(role="assistant", content="I'll generate that image"),
        tool_message_params=lambda x: [
            BaseMessageParam(role="assistant", content="Tool called"),
            BaseMessageParam(role="tool", content="http://example.com/image.jpg")
        ]
    )
    
    followup_response = Mock(
        content="Here's your sunset image!",
        tool=None,
        message_param=BaseMessageParam(role="assistant", content="Here's your sunset image!")
    )
    
    # Mock the tool call and provider
    with patch.object(ImageGenerationTool, 'call', new_callable=AsyncMock) as mock_call, \
         patch.object(ChatAgent, '_call_openpipe', side_effect=[mock_response, followup_response]):
        
        mock_call.return_value = "http://example.com/image.jpg"
        response = await agent._step(message)
        
        assert isinstance(response, str)
        assert "http://example.com/image.jpg" in response
        assert "![generated image]" in response
        assert len(agent.history) > 3  # system + user + initial response + tool response + followup

@pytest.mark.asyncio
async def test_chat_agent_image_generation_error():
    """Test the ChatAgent's handling of image generation errors."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    message = "Generate an image of a sunset"
    
    # Create a mock response object
    error_response = Mock()
    error_response.content = "I apologize, but I couldn't generate that image due to content policy restrictions. Perhaps we could try a different prompt?"
    error_response.tool = None
    
    # Mock both the tool error and the LLM response
    with patch.object(ImageGenerationTool, 'call', new_callable=AsyncMock) as mock_call, \
         patch.object(ChatAgent, '_call_openpipe', return_value=error_response):
        
        # Set up the tool mock to raise an error
        mock_call.side_effect = Exception("content_policy_violation: Not allowed")
        
        # Get the response
        response = await agent._step(message)
        
        # Verify the response
        assert isinstance(response, str)
        assert "couldn't generate" in response.lower() or "policy" in response.lower()
        assert len(agent.history) > 2  # system + user + error response

@pytest.mark.asyncio
async def test_chat_agent_conversation_flow():
    """Test a full conversation flow with the ChatAgent."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    
    # First message
    response1 = await agent._step("Hello!")
    assert isinstance(response1, str)
    assert len(agent.history) == 3
    
    # Follow-up message
    response2 = await agent._step("How are you?")
    assert isinstance(response2, str)
    assert len(agent.history) == 5  # Previous 3 + new user + assistant
    assert response2 != response1  # Responses should be different

@pytest.mark.asyncio
async def test_chat_agent_error_handling():
    """Test ChatAgent's general error handling."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    
    # Mock the provider call to raise an error
    with patch.object(ChatAgent, '_call_openpipe', side_effect=Exception("Test error")):
        response = await agent._step("Hello")
        assert "I apologize" in response
        assert "Test error" in response
        assert len(agent.history) >= 2  # system + error response

@pytest.mark.skipif(not os.getenv("RUN_FUNCTIONAL_TESTS"), reason="Functional tests are disabled")
@pytest.mark.asyncio
async def test_functional_providers():
    """Functional test for each provider with real API calls."""
    test_message = "What is 2+2?"
    
    for provider in ["openai", "anthropic", "openpipe"]:
        agent = ChatAgent(provider=provider)
        response = await agent._step(test_message)
        assert "4" in response.lower()  # Basic sanity check
        assert len(agent.history) == 3  # system + user + assistant 

@pytest.mark.asyncio
async def test_message_history_management():
    """Test that message history is managed correctly."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    
    # Check initial state (should only have system prompt)
    assert len(agent.history) == 1
    assert agent.history[0].role == "system"
    
    # Create mock response
    mock_response = Mock(
        content="Hello! How can I help you?",
        tool=None,
        message_param=BaseMessageParam(role="assistant", content="Hello! How can I help you?")
    )
    
    # Test adding a user message
    with patch.object(ChatAgent, '_call_openpipe', return_value=mock_response):
        await agent._step("Hello!")
        assert len(agent.history) == 3  # system + user + assistant
        assert agent.history[1].role == "user"
        assert agent.history[1].content == "Hello!"
        assert agent.history[2].role == "assistant"
        assert agent.history[2].content == "Hello! How can I help you?"

@pytest.mark.asyncio
async def test_message_history_with_tools():
    """Test message history when using tools."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    message = "Generate an image of a sunset"
    
    # Create mock responses
    tool_instance = ImageGenerationTool(prompt="sunset")
    initial_response = Mock(
        content="I'll generate that image",
        tool=tool_instance,
        message_param=BaseMessageParam(role="assistant", content="I'll generate that image"),
        tool_message_params=lambda x: [
            BaseMessageParam(role="assistant", content="Tool called"),
            BaseMessageParam(role="tool", content="http://example.com/image.jpg")
        ]
    )
    followup_response = Mock(
        content="Here's your sunset image",
        tool=None,
        message_param=BaseMessageParam(role="assistant", content="Here's your sunset image")
    )
    
    # Mock both the provider and tool calls
    with patch.object(ImageGenerationTool, 'call', new_callable=AsyncMock) as mock_tool, \
         patch.object(ChatAgent, '_call_openpipe', side_effect=[initial_response, followup_response]):
        
        mock_tool.return_value = "http://example.com/image.jpg"
        await agent._step(message)
        
        # Check history structure
        assert len(agent.history) >= 4  # system + user + initial response + tool response + followup
        assert agent.history[1].role == "user"
        assert agent.history[1].content == message
        assert agent.history[2].role == "assistant"  # Initial response about generating
        assert agent.history[2].content == "I'll generate that image"
        assert agent.history[3].role == "assistant"  # Tool response
        assert "Tool called" in agent.history[3].content

@pytest.mark.asyncio
async def test_message_history_error_handling():
    """Test message history management during errors."""
    agent = ChatAgent(provider="openpipe", persona=AUG_E)
    
    # Mock the provider to raise an error
    with patch.object(ChatAgent, '_call_openpipe', side_effect=Exception("Test error")):
        response = await agent._step("Hello")
        
        # Even with error, user message should be in history
        assert len(agent.history) >= 2  # system + user (error response not added)
        assert agent.history[1].role == "user"
        assert agent.history[1].content == "Hello" 