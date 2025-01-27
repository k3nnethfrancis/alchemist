"""Tests for base Graph functionality.

This module tests the core Graph class functionality including:
- Graph initialization and configuration
- Node registration and validation
- Edge creation and validation
- Entry point management
- Workflow execution
- Error handling
"""

import pytest
from typing import Dict, Any, Optional, List
import asyncio

from alchemist.ai.graph.base import Graph
from alchemist.ai.graph.nodes.base import Node
from alchemist.ai.graph.state import NodeState, NodeStatus
from pydantic import Field


class SimpleTestNode(Node):
    """Simple test node for graph testing."""
    
    id: str = Field(description="Node identifier")
    delay: float = Field(default=0.1, description="Processing delay in seconds")
    processed: bool = Field(default=False, description="Whether the node has been processed")
    next_nodes: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {"default": None, "error": None}
    )
    
    async def process(self, state: NodeState) -> Optional[str]:
        await asyncio.sleep(self.delay)
        self.processed = True
        state.results[self.id] = {"processed": True}
        return self.get_next_node()


class ErrorNode(Node):
    """Node that raises an error during processing."""
    
    id: str = Field(description="Node identifier")
    next_nodes: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {"default": None, "error": None}
    )
    
    async def process(self, state: NodeState) -> Optional[str]:
        raise ValueError("Test error")


@pytest.fixture
def simple_graph() -> Graph:
    """Fixture providing a basic graph."""
    return Graph()


@pytest.fixture
def test_node() -> SimpleTestNode:
    """Fixture providing a test node."""
    return SimpleTestNode(id="test_node")


class TestGraphInitialization:
    """Test suite for graph initialization."""

    def test_graph_init(self, simple_graph: Graph):
        """Test basic graph initialization."""
        assert isinstance(simple_graph.nodes, dict)
        assert isinstance(simple_graph.entry_points, dict)

    def test_graph_with_config(self):
        """Test graph initialization with config."""
        config = {"max_parallel": 5, "timeout": 30}
        graph = Graph(config=config)
        assert graph.config["max_parallel"] == 5
        assert graph.config["timeout"] == 30


class TestNodeManagement:
    """Test suite for node management."""

    def test_add_node(self, simple_graph: Graph, test_node: SimpleTestNode):
        """Test adding a node to the graph."""
        simple_graph.add_node(test_node)
        assert test_node.id in simple_graph.nodes
        assert simple_graph.nodes[test_node.id] == test_node

    def test_add_duplicate_node(self, simple_graph: Graph):
        """Test adding a duplicate node."""
        node1 = SimpleTestNode(id="test")
        node2 = SimpleTestNode(id="test")
        
        simple_graph.add_node(node1)
        with pytest.raises(ValueError):
            simple_graph.add_node(node2)

    def test_get_node(self, simple_graph: Graph, test_node: SimpleTestNode):
        """Test retrieving a node."""
        simple_graph.add_node(test_node)
        retrieved = simple_graph.get_node(test_node.id)
        assert retrieved == test_node

    def test_get_nonexistent_node(self, simple_graph: Graph):
        """Test retrieving a nonexistent node."""
        with pytest.raises(KeyError):
            simple_graph.get_node("nonexistent")


class TestEdgeManagement:
    """Test suite for edge management."""

    def test_add_edge(self, simple_graph: Graph):
        """Test adding an edge between nodes."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, node2.id)
        
        assert node2.id == node1.get_next_node()

    def test_add_edge_with_condition(self, simple_graph: Graph):
        """Test adding an edge with a condition."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, node2.id, condition="success")
        
        assert node2.id == node1.get_next_node("success")

    def test_add_invalid_edge(self, simple_graph: Graph, test_node: SimpleTestNode):
        """Test adding an edge with invalid nodes."""
        simple_graph.add_node(test_node)
        with pytest.raises(KeyError):
            simple_graph.add_edge(test_node.id, "nonexistent")


class TestEntryPoints:
    """Test suite for entry point management."""

    def test_add_entry_point(self, simple_graph: Graph, test_node: SimpleTestNode):
        """Test adding an entry point."""
        simple_graph.add_node(test_node)
        simple_graph.add_entry_point("start", test_node.id)
        assert "start" in simple_graph.entry_points
        assert simple_graph.entry_points["start"] == test_node.id

    def test_add_invalid_entry_point(self, simple_graph: Graph):
        """Test adding an entry point with invalid node."""
        with pytest.raises(KeyError):
            simple_graph.add_entry_point("start", "nonexistent")

    def test_get_entry_point(self, simple_graph: Graph, test_node: SimpleTestNode):
        """Test retrieving an entry point."""
        simple_graph.add_node(test_node)
        simple_graph.add_entry_point("start", test_node.id)
        node = simple_graph.get_entry_point("start")
        assert node == test_node


class TestGraphExecution:
    """Test suite for graph execution."""

    async def test_basic_execution(self, simple_graph: Graph):
        """Test basic graph execution."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, node2.id)
        simple_graph.add_entry_point("start", node1.id)
        
        state = await simple_graph.run("start")
        assert node1.processed
        assert node2.processed
        assert state.results[node1.id]["processed"]
        assert state.results[node2.id]["processed"]

    async def test_parallel_execution(self, simple_graph: Graph):
        """Test parallel node execution."""
        nodes = [SimpleTestNode(id=f"node{i}") for i in range(3)]
        for node in nodes:
            simple_graph.add_node(node)
        
        # Connect nodes in parallel
        simple_graph.add_entry_point("start", nodes[0].id)
        simple_graph.add_edge(nodes[0].id, nodes[1].id, "path1")
        simple_graph.add_edge(nodes[0].id, nodes[2].id, "path2")
        
        state = await simple_graph.run("start")
        assert all(node.processed for node in nodes)
        assert all(state.results[node.id]["processed"] for node in nodes)

    async def test_error_handling(self, simple_graph: Graph):
        """Test error handling during execution."""
        node1 = SimpleTestNode(id="node1")
        error_node = ErrorNode(id="error_node")
        node3 = SimpleTestNode(id="node3")
        
        simple_graph.add_node(node1)
        simple_graph.add_node(error_node)
        simple_graph.add_node(node3)
        
        simple_graph.add_edge(node1.id, error_node.id)
        simple_graph.add_edge(error_node.id, node3.id)
        simple_graph.add_entry_point("start", node1.id)
        
        state = await simple_graph.run("start")
        assert node1.processed
        assert not node3.processed
        assert "error_node" in state.errors
        assert isinstance(state.errors["error_node"], ValueError)


class TestGraphValidation:
    """Test suite for graph validation."""

    def test_validate_graph(self, simple_graph: Graph):
        """Test graph validation."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, node2.id)
        simple_graph.add_entry_point("start", node1.id)
        
        assert simple_graph.validate() == simple_graph

    def test_validate_empty_graph(self, simple_graph: Graph):
        """Test validation of empty graph."""
        with pytest.raises(ValueError):
            simple_graph.validate()

    def test_validate_disconnected_graph(self, simple_graph: Graph):
        """Test validation of disconnected graph."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")  # Disconnected node
        
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_entry_point("start", node1.id)
        
        with pytest.raises(ValueError):
            simple_graph.validate()


class TestGraphVisualization:
    """Test suite for graph visualization."""

    def test_to_dot(self, simple_graph: Graph):
        """Test DOT format generation."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, node2.id)
        
        dot = simple_graph.to_dot()
        assert isinstance(dot, str)
        assert "digraph" in dot
        assert "node1" in dot
        assert "node2" in dot 