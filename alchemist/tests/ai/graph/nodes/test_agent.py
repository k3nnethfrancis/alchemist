"""Tests for AgentNode functionality.

This module tests the AgentNode class which handles:
- LLM integration via Mirascope
- Prompt templating
- System prompt configuration
- Conversation state management
"""

import pytest
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from mirascope.core import BaseMessageParam, prompt_template
from alchemist.ai.graph.nodes import AgentNode
from alchemist.ai.graph.state import NodeState
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.base.logging import AlchemistLoggingConfig, VerbosityLevel


@prompt_template()
def test_prompt(input_text: str) -> List[BaseMessageParam]:
    """Simple test prompt template."""
    return [BaseMessageParam(role="user", content=f"Summarize: {input_text}")]


@pytest.fixture
def test_persona() -> PersonaConfig:
    """Fixture providing a test persona configuration."""
    return PersonaConfig(
        id="test-agent-v1",
        name="Test Agent",
        nickname="Testy",
        bio="A test agent for unit testing",
        personality={
            "traits": {
                "neuroticism": 0.2,
                "extraversion": 0.5,
                "openness": 0.7,
                "agreeableness": 0.8,
                "conscientiousness": 0.9
            },
            "stats": {
                "intelligence": 0.8,
                "wisdom": 0.7,
                "charisma": 0.6
            }
        },
        lore=["Created for testing", "Helps validate agent functionality"],
        style={
            "all": ["Uses clear language", "Stays focused on testing"],
            "chat": ["Responds concisely", "Maintains test context"]
        }
    )


@pytest.fixture
def agent_node(test_persona: PersonaConfig) -> AgentNode:
    """Fixture providing a configured agent node."""
    return AgentNode(
        id="test_agent",
        prompt_template=test_prompt,
        system_prompt="You are a helpful assistant that summarizes text concisely.",
        persona=test_persona,
        input_map={"input_text": "data.text"},
        next_nodes={"default": "next_node", "error": "error_node"}
    )


class TestAgentNodeInitialization:
    """Test suite for agent node initialization."""

    def test_agent_node_init(self, agent_node: AgentNode):
        """Test basic agent node initialization."""
        assert agent_node.id == "test_agent"
        assert agent_node.system_prompt == "You are a helpful assistant that summarizes text concisely."
        assert agent_node.persona.name == "Test Agent"
        assert agent_node.input_map["input_text"] == "data.text"

    def test_agent_node_without_prompt(self, test_persona: PersonaConfig):
        """Test agent node initialization without prompt template."""
        with pytest.raises(ValueError):
            AgentNode(
                id="test",
                system_prompt="Test prompt",
                persona=test_persona
            )

    def test_agent_node_without_system_prompt(self, test_persona: PersonaConfig):
        """Test agent node initialization without system prompt."""
        node = AgentNode(
            id="test",
            prompt_template=test_prompt,
            persona=test_persona
        )
        assert node.system_prompt is not None  # Should use default from persona


class TestAgentNodeProcessing:
    """Test suite for agent node processing."""

    async def test_basic_processing(self, agent_node: AgentNode):
        """Test basic agent node processing."""
        state = NodeState()
        state.set_data("data.text", "The quick brown fox jumps over the lazy dog.")
        
        next_node = await agent_node.process(state)
        assert next_node == "next_node"
        assert "response" in state.results[agent_node.id]

    async def test_missing_input(self, agent_node: AgentNode):
        """Test processing with missing input."""
        state = NodeState()
        next_node = await agent_node.process(state)
        assert next_node == "error_node"
        assert "error" in state.errors

    async def test_conversation_history(self, agent_node: AgentNode):
        """Test conversation history management."""
        state = NodeState()
        state.set_data("data.text", "First message")
        await agent_node.process(state)
        
        state.set_data("data.text", "Second message")
        await agent_node.process(state)
        
        history = agent_node.get_history(state)
        assert len(history) > 0
        assert isinstance(history[0], BaseMessageParam)


class TestAgentNodeConfiguration:
    """Test suite for agent node configuration."""

    def test_persona_integration(self, agent_node: AgentNode):
        """Test persona integration."""
        assert agent_node.persona.id == "test-agent-v1"
        assert "traits" in agent_node.persona.personality
        assert "stats" in agent_node.persona.personality

    def test_prompt_configuration(self, agent_node: AgentNode):
        """Test prompt template configuration."""
        messages = test_prompt("test")
        assert isinstance(messages, list)
        assert all(isinstance(m, BaseMessageParam) for m in messages)

    def test_logging_configuration(self, agent_node: AgentNode):
        """Test logging configuration."""
        assert agent_node.logging_config is not None
        assert isinstance(agent_node.logging_config, AlchemistLoggingConfig) 