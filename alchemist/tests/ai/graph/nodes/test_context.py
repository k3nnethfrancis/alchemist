"""Tests for ContextNode functionality.

This module tests the ContextNode class which handles:
- Context data management
- External context source configuration
- State updates
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
        next_nodes={"default": "next_node", "error": "error_node"}
    )


class TestContextNodeInitialization:
    """Test suite for context node initialization."""

    def test_context_node_init(self, context_node: ContextNode):
        """Test basic context node initialization."""
        assert context_node.id == "test_context"
        assert context_node.context_source == "supabase"

    def test_context_node_without_maps(self):
        """Test context node initialization without maps."""
        node = ContextNode(id="test")
        assert isinstance(node.input_map, dict)
        assert node.context_source == "supabase"

    def test_context_node_validation(self):
        """Test context node validation."""
        node = ContextNode(
            id="test",
            context_source="redis"
        )
        assert node.context_source == "redis"


class TestContextNodeProcessing:
    """Test suite for context node processing."""

    async def test_basic_processing(self, context_node: ContextNode):
        """Test basic context node processing."""
        state = NodeState()
        next_node = await context_node.process(state)
        assert next_node == "next_node"
        assert state.get_data("external_context") == "Fetched from supabase ..."

    async def test_multiple_mappings(self):
        """Test processing with custom context source."""
        node = ContextNode(
            id="test",
            context_source="redis"
        )
        
        state = NodeState()
        await node.process(state)
        assert state.get_data("external_context") == "Fetched from redis ..."

    async def test_nested_mappings(self):
        """Test processing with state updates."""
        node = ContextNode(
            id="test"
        )
        
        state = NodeState()
        await node.process(state)
        assert "external_context" in state.data


class TestErrorHandling:
    """Test suite for error handling."""

    async def test_missing_input(self, context_node: ContextNode):
        """Test basic processing without input."""
        state = NodeState()
        next_node = await context_node.process(state)
        assert next_node == "next_node"

    async def test_invalid_mapping(self):
        """Test processing with default next node."""
        node = ContextNode(
            id="test"
        )
        
        state = NodeState()
        next_node = await node.process(state)
        assert next_node is None


class TestStateManagement:
    """Test suite for state management."""

    async def test_state_updates(self, context_node: ContextNode):
        """Test state updates during processing."""
        state = NodeState()
        await context_node.process(state)
        assert "external_context" in state.data

    async def test_result_storage(self, context_node: ContextNode):
        """Test context updates in state."""
        state = NodeState()
        await context_node.process(state)
        assert state.get_data("external_context") == "Fetched from supabase ..."

    async def test_metadata_handling(self):
        """Test metadata handling in state."""
        node = ContextNode(
            id="test",
            metadata={"description": "Test node"}
        )
        
        state = NodeState()
        await node.process(state)
        assert node.metadata["description"] == "Test node" 