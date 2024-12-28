"""Test suite for graph nodes."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.nodes.base import NodeState, NodeContext
from alchemist.ai.graph.nodes.decisions import BinaryDecisionNode, MultiChoiceNode
from alchemist.ai.graph.nodes.actions import ToolNode, WaitNode
from alchemist.ai.graph.nodes.responses import SimpleResponseNode, StructuredResponseNode, TemplatedResponseNode
from mirascope.core import BaseMessageParam

@pytest.fixture
def node_state():
    """Create a NodeState instance for testing."""
    return NodeState(
        context=NodeContext(metadata={}),
        results={},
        data={}
    )

@pytest.fixture
def test_agent():
    """Create a BaseAgent instance for testing."""
    return BaseAgent(provider="openai")

class TestDecisionNodes:
    """Test suite for decision nodes."""
    
    @pytest.mark.asyncio
    async def test_binary_decision(self, node_state: NodeState, test_agent: BaseAgent):
        """Test binary decision node."""
        node = BinaryDecisionNode(
            id="test_binary",
            prompt="Should we respond to this message?",
            next_nodes={
                "yes": "respond_node",
                "no": "wait_node"
            },
            agent=test_agent
        )
        
        # Mock LLM call
        with patch.object(BaseAgent, '_get_response', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.content = "yes"
            mock_llm.return_value = mock_response
            next_node = await node.process(node_state)
            assert next_node == "respond_node"
            
            mock_response = Mock()
            mock_response.content = "no"
            mock_llm.return_value = mock_response
            next_node = await node.process(node_state)
            assert next_node == "wait_node"
    
    @pytest.mark.asyncio
    async def test_multi_choice(self, node_state: NodeState, test_agent: BaseAgent):
        """Test multi-choice decision node."""
        node = MultiChoiceNode(
            id="test_multi",
            prompt="How should we respond?",
            choices=["casual", "formal", "ignore"],
            next_nodes={
                "casual": "casual_node",
                "formal": "formal_node",
                "ignore": "wait_node"
            },
            agent=test_agent
        )
        
        # Mock LLM call
        with patch.object(BaseAgent, '_get_response', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.content = "casual"
            mock_llm.return_value = mock_response
            next_node = await node.process(node_state)
            assert next_node == "casual_node"
            
            mock_response = Mock()
            mock_response.content = "formal"
            mock_llm.return_value = mock_response
            next_node = await node.process(node_state)
            assert next_node == "formal_node"
            
            mock_response = Mock()
            mock_response.content = "ignore"
            mock_llm.return_value = mock_response
            next_node = await node.process(node_state)
            assert next_node == "wait_node"

class TestActionNodes:
    """Test suite for action nodes."""
    
    @pytest.mark.asyncio
    async def test_tool_node(self, node_state: NodeState, test_agent: BaseAgent):
        """Test direct tool execution."""
        node = ToolNode(
            id="test_tool",
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
            next_nodes={"default": "next_node"},
            agent=test_agent
        )
        
        # Mock tool execution
        with patch('alchemist.ai.graph.nodes.actions.execute_tool', new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = "Tool result"
            next_node = await node.process(node_state)
            assert next_node == "next_node"
            
class TestResponseNodes:
    """Test suite for response nodes."""
    
    @pytest.mark.asyncio
    async def test_simple_response(self, node_state: NodeState, test_agent: BaseAgent):
        """Test simple response generation."""
        node = SimpleResponseNode(
            id="test_simple",
            prompt="Generate a greeting.",
            next_nodes={"default": "next_node"},
            agent=test_agent
        )
        
        # Mock LLM call
        with patch.object(BaseAgent, '_get_response', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.content = "Hello!"
            mock_llm.return_value = mock_response
            next_node = await node.process(node_state)
            assert next_node == "next_node"
            
    @pytest.mark.asyncio
    async def test_structured_response(self, node_state: NodeState, test_agent: BaseAgent):
        """Test structured response generation."""
        node = StructuredResponseNode(
            id="test_structured",
            prompt="Generate a greeting with metadata.",
            output_schema={
                "message": str,
                "metadata": Dict[str, Any]
            },
            next_nodes={"default": "next_node"},
            agent=test_agent
        )
        
        # Mock LLM call
        with patch.object(BaseAgent, '_get_response', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.content = '{"message": "Hello!", "metadata": {"tone": "friendly"}}'
            mock_llm.return_value = mock_response
            next_node = await node.process(node_state)
            assert next_node == "next_node"
            
    @pytest.mark.asyncio
    async def test_templated_response(self, node_state: NodeState, test_agent: BaseAgent):
        """Test templated response generation."""
        node = TemplatedResponseNode(
            id="test_template",
            prompt="Generate a greeting message.",
            template="Hello {name}! {content}",
            template_vars={"name": "User"},
            next_nodes={"default": "next_node"},
            agent=test_agent
        )
        
        # Mock LLM call
        with patch.object(BaseAgent, '_get_response', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.content = "Welcome to our service!"
            mock_llm.return_value = mock_response
            next_node = await node.process(node_state)
            assert next_node == "next_node" 