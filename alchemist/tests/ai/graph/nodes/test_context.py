"""Tests for ContextNode functionality.

This module tests the ContextNode class which handles:
- Context data management
- Input/output mapping
- State updates
- Error handling
"""

import pytest
from typing import Dict, Any, Optional

from alchemist.ai.graph.nodes import ContextNode
from alchemist.ai.graph.state import NodeState


@pytest.fixture
def context_node() -> ContextNode:
    """Fixture providing a basic context node."""
    return ContextNode(
        id="test_context",
        input_map={"value": "source.value"},
        output_map={"result": "target.value"},
        next_nodes={"default": "next_node", "error": "error_node"}
    )


class TestContextNodeInitialization:
    """Test suite for context node initialization."""

    def test_context_node_init(self, context_node: ContextNode):
        """Test basic context node initialization."""
        assert context_node.id == "test_context"
        assert context_node.input_map["value"] == "source.value"
        assert context_node.output_map["result"] == "target.value"

    def test_context_node_without_maps(self):
        """Test context node initialization without maps."""
        node = ContextNode(id="test")
        assert isinstance(node.input_map, dict)
        assert isinstance(node.output_map, dict)

    def test_context_node_validation(self):
        """Test context node validation."""
        with pytest.raises(ValueError):
            ContextNode(
                id="test",
                input_map={"": "invalid"},
                output_map={"result": ""}
            )


class TestContextNodeProcessing:
    """Test suite for context node processing."""

    async def test_basic_processing(self, context_node: ContextNode):
        """Test basic context node processing."""
        state = NodeState()
        state.set_data("source.value", "test_value")
        
        next_node = await context_node.process(state)
        assert next_node == "next_node"
        assert state.get_data("target.value") == "test_value"

    async def test_multiple_mappings(self):
        """Test processing with multiple mappings."""
        node = ContextNode(
            id="test",
            input_map={
                "value1": "source.v1",
                "value2": "source.v2"
            },
            output_map={
                "result1": "target.r1",
                "result2": "target.r2"
            }
        )
        
        state = NodeState()
        state.set_data("source.v1", "value1")
        state.set_data("source.v2", "value2")
        
        await node.process(state)
        assert state.get_data("target.r1") == "value1"
        assert state.get_data("target.r2") == "value2"

    async def test_nested_mappings(self):
        """Test processing with nested mappings."""
        node = ContextNode(
            id="test",
            input_map={"value": "source.nested.value"},
            output_map={"result": "target.nested.result"}
        )
        
        state = NodeState()
        state.set_data("source.nested.value", "test_value")
        
        await node.process(state)
        assert state.get_data("target.nested.result") == "test_value"


class TestErrorHandling:
    """Test suite for error handling."""

    async def test_missing_input(self, context_node: ContextNode):
        """Test handling of missing input values."""
        state = NodeState()
        next_node = await context_node.process(state)
        assert next_node == "error_node"
        assert "test_context" in state.errors

    async def test_invalid_mapping(self):
        """Test handling of invalid mappings."""
        node = ContextNode(
            id="test",
            input_map={"value": "invalid..path"},
            next_nodes={"error": "error_node"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        assert next_node == "error_node"
        assert "test" in state.errors


class TestStateManagement:
    """Test suite for state management."""

    async def test_state_updates(self, context_node: ContextNode):
        """Test state updates during processing."""
        state = NodeState()
        state.set_data("source.value", "initial")
        
        await context_node.process(state)
        
        # Check input data is preserved
        assert state.get_data("source.value") == "initial"
        # Check output data is set
        assert state.get_data("target.value") == "initial"

    async def test_result_storage(self, context_node: ContextNode):
        """Test result storage in state."""
        state = NodeState()
        state.set_data("source.value", "test_value")
        
        await context_node.process(state)
        assert context_node.id in state.results
        assert "mapped" in state.results[context_node.id]

    async def test_metadata_handling(self):
        """Test metadata handling in state."""
        node = ContextNode(
            id="test",
            metadata={"description": "Test node"}
        )
        
        state = NodeState()
        await node.process(state)
        assert node.metadata["description"] == "Test node" 