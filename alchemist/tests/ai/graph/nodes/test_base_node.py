"""Tests for the abstract base Node class."""

import pytest
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.state import NodeStateProtocol, NodeState
from typing import Optional

class MinimalNode(Node):
    """A minimal concrete subclass implementing the abstract process()."""

    async def process(self, state: NodeStateProtocol) -> Optional[str]:
        return None

def test_node_validation():
    """Test that Node validation fails if 'id' is empty."""
    with pytest.raises(ValueError):
        MinimalNode(id="  ")  # invalid (only whitespace)

def test_node_initialization():
    """Test that Node can initialize with a valid ID."""
    node = MinimalNode(id="test_node")
    assert node.id == "test_node"
    assert node.validate() is True

@pytest.mark.asyncio
async def test_node_process_not_implemented():
    """Test that the Node base class raises NotImplementedError if not overridden."""

    class NotImplementedNode(Node):
        async def process(self, state: NodeStateProtocol) -> Optional[str]:
            return await super().process(state)  # calls base -> NotImplementedError

    not_impl_node = NotImplementedNode(id="no_process")
    with pytest.raises(NotImplementedError):
        await not_impl_node.process(NodeState()) 