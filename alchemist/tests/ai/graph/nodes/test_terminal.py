"""Tests for TerminalNode functionality.

This module tests the TerminalNode class which handles:
- Terminal command execution
- Input/output mapping
- Error handling
- State management
"""

import pytest
from typing import Dict, Any, Optional
import asyncio

from alchemist.ai.graph.nodes import TerminalNode
from alchemist.ai.graph.state import NodeState


@pytest.fixture
def terminal_node() -> TerminalNode:
    """Fixture providing a basic terminal node."""
    return TerminalNode(
        id="test_terminal",
        command="echo 'test'",
        next_nodes={"default": "next_node", "error": "error_node"}
    )


class TestTerminalNodeInitialization:
    """Test suite for terminal node initialization."""

    def test_terminal_node_init(self, terminal_node: TerminalNode):
        """Test basic terminal node initialization."""
        assert terminal_node.id == "test_terminal"
        assert terminal_node.command == "echo 'test'"
        assert "default" in terminal_node.next_nodes
        assert "error" in terminal_node.next_nodes

    def test_terminal_node_without_command(self):
        """Test terminal node initialization without command."""
        with pytest.raises(ValueError):
            TerminalNode(id="test")

    def test_terminal_node_with_invalid_command(self):
        """Test terminal node initialization with invalid command."""
        with pytest.raises(ValueError):
            TerminalNode(id="test", command="")


class TestTerminalNodeProcessing:
    """Test suite for terminal node processing."""

    async def test_basic_processing(self, terminal_node: TerminalNode):
        """Test basic terminal node processing."""
        state = NodeState()
        next_node = await terminal_node.process(state)
        
        assert next_node == "next_node"
        assert terminal_node.id in state.results
        assert "output" in state.results[terminal_node.id]
        assert "test" in state.results[terminal_node.id]["output"]

    async def test_command_with_variables(self):
        """Test command execution with variables."""
        node = TerminalNode(
            id="test",
            command="echo '${message}'",
            input_map={"message": "data.message"}
        )
        
        state = NodeState()
        state.set_data("data.message", "hello world")
        
        await node.process(state)
        assert "hello world" in state.results[node.id]["output"]

    async def test_command_timeout(self):
        """Test command execution timeout."""
        node = TerminalNode(
            id="test",
            command="sleep 5",  # Command that takes too long
            timeout=0.1,
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        
        assert next_node == "error_node"
        assert "test" in state.errors
        assert "timeout" in str(state.errors["test"]).lower()


class TestErrorHandling:
    """Test suite for error handling."""

    async def test_invalid_command(self):
        """Test handling of invalid command execution."""
        node = TerminalNode(
            id="test",
            command="nonexistent_command",
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        
        assert next_node == "error_node"
        assert "test" in state.errors

    async def test_missing_variable(self):
        """Test handling of missing variable in command."""
        node = TerminalNode(
            id="test",
            command="echo '${missing}'",
            input_map={"missing": "data.missing"},
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        
        assert next_node == "error_node"
        assert "test" in state.errors


class TestStateManagement:
    """Test suite for state management."""

    async def test_result_storage(self, terminal_node: TerminalNode):
        """Test storage of command results in state."""
        state = NodeState()
        await terminal_node.process(state)
        
        assert terminal_node.id in state.results
        assert "output" in state.results[terminal_node.id]
        assert "exit_code" in state.results[terminal_node.id]

    async def test_error_storage(self):
        """Test storage of command errors in state."""
        node = TerminalNode(
            id="test",
            command="exit 1",  # Command that fails
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        await node.process(state)
        
        assert "test" in state.errors
        assert state.results["test"]["exit_code"] == 1

    async def test_environment_variables(self):
        """Test command execution with environment variables."""
        node = TerminalNode(
            id="test",
            command="echo $TEST_VAR",
            env={"TEST_VAR": "test_value"}
        )
        
        state = NodeState()
        await node.process(state)
        
        assert "test_value" in state.results[node.id]["output"] 