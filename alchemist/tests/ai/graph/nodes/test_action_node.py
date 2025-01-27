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
    
    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """Simulate tool execution."""
        return {"result": "mock_result"}


class MockToolConfig(BaseModel):
    """Mock tool configuration."""
    param1: str
    param2: int = 42


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
            "param1": "input.param1",
            "param2": "input.param2"
        }
    )


class TestActionNodeInitialization:
    """Test suite for action node initialization."""

    def test_action_node_init(self, action_node: ActionNode):
        """Test basic action node initialization."""
        assert action_node.id == "test_action"
        assert action_node.tool is not None
        assert action_node.input_map == {
            "param1": "input.param1",
            "param2": "input.param2"
        }


class TestActionNodeExecution:
    """Test suite for action node execution."""

    async def test_basic_execution(self, action_node: ActionNode):
        """Test basic tool execution through action node."""
        state = NodeState()
        state.set_data("input", {"param1": "test", "param2": 42})
        
        result_state = await action_node.process(state)
        assert result_state.results[action_node.id] == {"result": "mock_result"}

    async def test_missing_input(self, action_node: ActionNode):
        """Test execution with missing required input."""
        state = NodeState()
        with pytest.raises(KeyError):
            await action_node.process(state)


class TestActionNodeValidation:
    """Test suite for action node input validation."""

    async def test_input_validation(self, mock_tool: MockTool):
        """Test input validation with Pydantic model."""
        node = ActionNode(
            id="validated_action",
            tool=mock_tool,
            input_map={"param1": "input.param1", "param2": "input.param2"},
            input_model=MockToolConfig
        )
        
        # Valid input
        state = NodeState()
        state.set_data("input", {"param1": "test", "param2": 42})
        await node.process(state)
        
        # Invalid input
        state = NodeState()
        state.set_data("input", {"param1": "test", "param2": "not_an_int"})
        with pytest.raises(ValueError):
            await node.process(state)


class TestActionNodeEvents:
    """Test suite for action node event emission."""

    async def test_execution_events(self, action_node: ActionNode):
        """Test that action node emits appropriate events during execution."""
        events = []
        
        def event_handler(event: Dict[str, Any]):
            events.append(event)
            
        action_node.add_event_handler(event_handler)
        
        state = NodeState()
        state.set_data("input", {"param1": "test", "param2": 42})
        
        await action_node.process(state)
        
        event_types = [e["type"] for e in events]
        assert "STARTED" in event_types
        assert "TOOL_CALLED" in event_types
        assert "TOOL_COMPLETED" in event_types
        assert "COMPLETED" in event_types
