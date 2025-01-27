"""Tests for ActionNode functionality.

This module tests the ActionNode class which handles:
- Tool execution
- Input/output mapping
- Error handling
- State management
"""

import pytest
from typing import Dict, Any, Optional
from pydantic import BaseModel

from mirascope.core import BaseTool
from alchemist.ai.graph.nodes import ActionNode
from alchemist.ai.graph.state import NodeState


class MockTool(BaseTool):
    """Mock tool for testing action node functionality."""
    
    def __init__(self):
        self.called = False
        self.args = {}
    
    @classmethod
    def _name(cls) -> str:
        return "mock_tool"
        
    async def call(self) -> Dict[str, Any]:
        """Simulate tool execution."""
        self.called = True
        return {"result": "mock_result"}


class FailingTool(BaseTool):
    """Tool that raises an error during execution."""
    
    @classmethod
    def _name(cls) -> str:
        return "failing_tool"
        
    async def call(self) -> Dict[str, Any]:
        """Simulate tool failure."""
        raise ValueError("Tool execution failed")


@pytest.fixture
def mock_tool() -> MockTool:
    """Fixture providing a mock tool."""
    return MockTool()


@pytest.fixture
def failing_tool() -> FailingTool:
    """Fixture providing a failing tool."""
    return FailingTool()


@pytest.fixture
def action_node(mock_tool: MockTool) -> ActionNode:
    """Fixture providing a configured action node."""
    return ActionNode(
        id="test_action",
        tool=mock_tool,
        input_map={"value": "data.input_value"},
        next_nodes={"default": "next_node", "error": "error_node"}
    )


class TestActionNodeInitialization:
    """Test suite for action node initialization."""

    def test_action_node_init(self, action_node: ActionNode):
        """Test basic action node initialization."""
        assert action_node.id == "test_action"
        assert isinstance(action_node.tool, BaseTool)
        assert action_node.input_map["value"] == "data.input_value"

    def test_action_node_without_tool(self):
        """Test action node initialization without tool."""
        with pytest.raises(ValueError):
            ActionNode(id="test")

    def test_action_node_with_invalid_tool(self):
        """Test action node initialization with invalid tool."""
        class InvalidTool:
            pass
        
        with pytest.raises(TypeError):
            ActionNode(id="test", tool=InvalidTool())


class TestActionNodeProcessing:
    """Test suite for action node processing."""

    async def test_basic_processing(self, action_node: ActionNode):
        """Test basic action node processing."""
        state = NodeState()
        state.set_data("data.input_value", "test_input")
        
        next_node = await action_node.process(state)
        assert next_node == "next_node"
        assert state.results[action_node.id]["result"] == "mock_result"

    async def test_tool_execution(self, action_node: ActionNode):
        """Test tool execution during processing."""
        state = NodeState()
        await action_node.process(state)
        assert action_node.tool.called

    async def test_tool_failure(self, failing_tool: FailingTool):
        """Test handling of tool execution failure."""
        node = ActionNode(
            id="failing_action",
            tool=failing_tool,
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        assert next_node == "error_node"
        assert "failing_action" in state.errors


class TestInputMapping:
    """Test suite for input mapping."""

    async def test_input_mapping(self, action_node: ActionNode):
        """Test input mapping to tool parameters."""
        state = NodeState()
        state.set_data("data.input_value", "mapped_value")
        
        await action_node.process(state)
        assert action_node.tool.args.get("value") == "mapped_value"

    async def test_missing_input(self, action_node: ActionNode):
        """Test handling of missing input values."""
        state = NodeState()
        next_node = await action_node.process(state)
        assert next_node == "error_node"
        assert "test_action" in state.errors


class TestStateManagement:
    """Test suite for state management."""

    async def test_result_storage(self, action_node: ActionNode):
        """Test storage of tool results in state."""
        state = NodeState()
        await action_node.process(state)
        
        assert action_node.id in state.results
        assert "result" in state.results[action_node.id]

    async def test_error_storage(self, failing_tool: FailingTool):
        """Test storage of errors in state."""
        node = ActionNode(
            id="failing_action",
            tool=failing_tool
        )
        
        state = NodeState()
        await node.process(state)
        
        assert "failing_action" in state.errors
        assert isinstance(state.errors["failing_action"], ValueError) 