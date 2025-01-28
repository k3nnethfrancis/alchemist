"""Tests for TerminalNode functionality.

This module tests the TerminalNode class which handles:
- Workflow termination
- Status marking
- State management
"""

import pytest
from typing import Dict, Any, Optional

from alchemist.ai.graph.nodes import TerminalNode
from alchemist.ai.graph.state import NodeState, NodeStatus


@pytest.fixture
def terminal_node() -> TerminalNode:
    """Fixture providing a basic terminal node."""
    return TerminalNode(
        id="test_terminal",
        next_nodes={"default": "next_node", "error": "error_node"}
    )


class TestTerminalNodeInitialization:
    """Test suite for terminal node initialization."""

    def test_terminal_node_init(self, terminal_node: TerminalNode):
        """Test basic terminal node initialization."""
        assert terminal_node.id == "test_terminal"
        assert "default" in terminal_node.next_nodes
        assert "error" in terminal_node.next_nodes

    def test_terminal_node_without_command(self):
        """Test terminal node initialization without command."""
        node = TerminalNode(id="test")
        assert node.id == "test"

    def test_terminal_node_with_invalid_command(self):
        """Test terminal node initialization with invalid command."""
        node = TerminalNode(id="test", next_nodes={"default": None})
        assert node.next_nodes["default"] is None


class TestTerminalNodeProcessing:
    """Test suite for terminal node processing."""

    async def test_basic_processing(self, terminal_node: TerminalNode):
        """Test basic terminal node processing."""
        state = NodeState()
        next_node = await terminal_node.process(state)
        
        assert next_node is None
        assert state.status[terminal_node.id] == NodeStatus.TERMINAL

    async def test_command_with_variables(self):
        """Test terminal node with input mapping."""
        node = TerminalNode(
            id="test",
            input_map={"message": "data.message"}
        )
        
        state = NodeState()
        state.set_data("data.message", "hello world")
        
        next_node = await node.process(state)
        assert next_node is None
        assert state.status[node.id] == NodeStatus.TERMINAL

    async def test_command_timeout(self):
        """Test terminal node with timeout."""
        node = TerminalNode(
            id="test",
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        
        assert next_node is None
        assert state.status[node.id] == NodeStatus.TERMINAL


class TestErrorHandling:
    """Test suite for error handling."""

    async def test_invalid_command(self):
        """Test terminal node with error next node."""
        node = TerminalNode(
            id="test",
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        
        assert next_node is None
        assert state.status[node.id] == NodeStatus.TERMINAL

    async def test_missing_variable(self):
        """Test terminal node with missing input."""
        node = TerminalNode(
            id="test",
            input_map={"missing": "data.missing"},
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        
        assert next_node is None
        assert state.status[node.id] == NodeStatus.TERMINAL


class TestStateManagement:
    """Test suite for state management."""

    async def test_result_storage(self, terminal_node: TerminalNode):
        """Test terminal node status in state."""
        state = NodeState()
        await terminal_node.process(state)
        
        assert state.status[terminal_node.id] == NodeStatus.TERMINAL

    async def test_error_storage(self):
        """Test terminal node with error handling."""
        node = TerminalNode(
            id="test",
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        await node.process(state)
        
        assert state.status[node.id] == NodeStatus.TERMINAL

    async def test_environment_variables(self):
        """Test terminal node with environment variables."""
        node = TerminalNode(
            id="test",
            metadata={"env": {"TEST_VAR": "test_value"}}
        )
        
        state = NodeState()
        await node.process(state)
        
        assert state.status[node.id] == NodeStatus.TERMINAL 