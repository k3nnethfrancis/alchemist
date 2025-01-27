"""Tests for ActionNode functionality.

This module contains tests for the ActionNode class, which handles tool execution
and state management in the graph system.
"""

import pytest
from typing import Dict, Any
from pydantic import BaseModel

from alchemist.ai.graph.nodes.actions import ActionNode
from alchemist.ai.graph.state import NodeState


class MockTool:
    """Mock tool for testing action node functionality."""
    
    def __call__(self, param1: str, param2: int = 42) -> Dict[str, Any]:
        """Simulate tool execution."""
        return {"result": f"{param1}_{param2}"}


@pytest.fixture
def mock_tool() -> MockTool:
    """Fixture providing a mock tool."""
    return MockTool()


@pytest.fixture
def action_node(mock_tool: MockTool) -> ActionNode:
    """Fixture providing a basic action node with mock tool."""
    return ActionNode(
        id="test_action",
        tool=mock_tool,
        input_map={
            "param1": "data.input.param1",
            "param2": "data.input.param2"
        },
        next_nodes={"default": "next_node", "error": "error_node"}
    )


class TestActionNodeInitialization:
    """Test suite for action node initialization."""

    def test_action_node_init(self, action_node: ActionNode):
        """Test basic action node initialization."""
        assert action_node.id == "test_action"
        assert callable(action_node.tool)
        assert action_node.input_map == {
            "param1": "data.input.param1",
            "param2": "data.input.param2"
        }

    def test_action_node_with_invalid_tool(self):
        """Test action node initialization with invalid tool."""
        class InvalidTool:
            def __init__(self):
                pass  # Not callable
        
        with pytest.raises(ValueError):
            ActionNode(id="test", tool=InvalidTool())


class TestActionNodeExecution:
    """Test suite for action node execution."""

    @pytest.mark.asyncio
    async def test_basic_execution(self, action_node: ActionNode):
        """Test basic tool execution through action node."""
        state = NodeState()
        state.data["input"] = {"param1": "test", "param2": 42}
        
        next_node = await action_node.process(state)
        assert next_node == "next_node"
        result = state.results[action_node.id]
        assert "result" in result
        assert result["result"] == {"result": "test_42"}
        assert "timing" in result

    @pytest.mark.asyncio
    async def test_missing_input(self, action_node: ActionNode):
        """Test execution with missing required input."""
        state = NodeState()
        next_node = await action_node.process(state)
        assert next_node == "error_node"
        assert "test_action" in state.errors
        assert "Key 'input' not found while traversing" in state.errors["test_action"]


class TestStateManagement:
    """Test suite for state management."""

    @pytest.mark.asyncio
    async def test_required_state(self, mock_tool: MockTool):
        """Test required state validation."""
        node = ActionNode(
            id="test",
            tool=mock_tool,
            required_state=["required_key"],
            input_map={
                "param1": "data.input.param1",
                "param2": "data.input.param2"
            },
            next_nodes={"default": "next", "error": "error"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        assert next_node == "error"
        assert "test" in state.errors
        assert "Missing required state keys" in state.errors["test"]
        
        # Add required state and input data
        state.data["required_key"] = "value"
        state.data["input"] = {"param1": "test", "param2": 42}
        next_node = await node.process(state)
        assert next_node == "next"
        assert "test" in state.results

    @pytest.mark.asyncio
    async def test_preserve_state(self, mock_tool: MockTool):
        """Test state preservation."""
        node = ActionNode(
            id="test",
            tool=mock_tool,
            preserve_state=["keep_this"],
            input_map={
                "param1": "data.input.param1",
                "param2": "data.input.param2"
            },
            next_nodes={"default": "next"}
        )
        
        state = NodeState()
        state.results["keep_this"] = "preserved"
        state.results["remove_this"] = "temporary"
        state.data["input"] = {"param1": "test", "param2": 42}
        
        await node.process(state)
        assert "keep_this" in state.results
        assert "remove_this" not in state.results
        assert node.id in state.results
