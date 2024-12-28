"""Tests for graph node functionality."""

import pytest
from alchemist.ai.graph.base import Graph, NodeState, NodeContext
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.graph.nodes.decisions import BinaryDecisionNode
from alchemist.ai.base.agent import BaseAgent
from typing import Optional
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def test_agent():
    """Create a test agent."""
    agent = MagicMock(spec=BaseAgent)
    agent.get_response = AsyncMock(return_value="Test response")
    return agent

@pytest.fixture
def test_state():
    """Create a test state."""
    return NodeState(
        context=NodeContext(),
        results={"input": {"message": "test message"}},
        data={"step_number": 1}
    )

class TestLLMNode:
    """Test LLM node functionality."""
    
    async def test_basic_processing(self, test_agent, test_state):
        """Test basic LLM node processing."""
        node = LLMNode(
            id="test_node",
            agent=test_agent,
            prompt="Process this: {input[message]}",
            next_nodes={"default": None}
        )
        
        result = await node.process(test_state)
        
        assert result is None  # End node
        assert "test_node" in test_state.results
        assert "response" in test_state.results["test_node"]

    async def test_error_handling(self, test_agent, test_state):
        """Test LLM node error handling."""
        # Mock agent to raise an exception
        test_agent.get_response.side_effect = Exception("Test error")
        
        node = LLMNode(
            id="test_node",
            agent=test_agent,
            prompt="Test prompt",
            next_nodes={"default": None}
        )
        
        result = await node.process(test_state)
        assert result is None  # Base class returns None on error
        assert "test_node" in test_state.results
        assert "error" in test_state.results["test_node"] 