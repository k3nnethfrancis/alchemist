"""Graph System Base Tests

Tests the core functionality of the graph system including:
1. Node execution
2. State management
3. Error handling
"""

import pytest
from typing import Optional, Dict, Any
from pydantic import Field

from alchemist.ai.graph.base import Node, Graph, NodeState, NodeContext

# Test Nodes

class CounterNode(Node):
    """Test node that counts executions."""
    
    count: int = Field(default=0)
    
    async def process(self, state: NodeState) -> Optional[str]:
        self.count += 1
        state.results[self.id] = {"count": self.count}
        return self.next_nodes.get("default")

class ErrorNode(Node):
    """Test node that raises an error."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        raise ValueError("Test error")

class DataNode(Node):
    """Test node that modifies state data."""
    
    key: str
    value: Any
    
    async def process(self, state: NodeState) -> Optional[str]:
        state.data[self.key] = self.value
        return self.next_nodes.get("default")

# Tests

@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    graph = Graph()
    
    # Add nodes
    start = CounterNode(id="start")
    middle = DataNode(id="middle", key="test", value="value")
    end = CounterNode(id="end")
    
    graph.add_node(start)
    graph.add_node(middle)
    graph.add_node(end)
    
    # Add edges
    graph.add_edge("start", "default", "middle")
    graph.add_edge("middle", "default", "end")
    
    # Add entry point
    graph.add_entry_point("main", "start")
    
    return graph

@pytest.mark.asyncio
async def test_graph_execution(simple_graph):
    """Test basic graph execution flow."""
    state = await simple_graph.run("main")
    
    # Check node execution
    assert simple_graph.nodes["start"].count == 1
    assert simple_graph.nodes["end"].count == 1
    
    # Check state updates
    assert state.data["test"] == "value"
    assert state.results["start"]["count"] == 1
    assert state.results["end"]["count"] == 1

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in graph execution."""
    graph = Graph()
    
    # Create error path
    start = ErrorNode(id="start")
    fallback = CounterNode(id="fallback")
    
    graph.add_node(start)
    graph.add_node(fallback)
    
    # Add error edge
    graph.add_edge("start", "error", "fallback")
    
    # Add entry point
    graph.add_entry_point("main", "start")
    
    # Run graph
    state = await graph.run("main")
    
    # Check error handling
    assert "error" in state.results["start"]
    assert state.results["fallback"]["count"] == 1

@pytest.mark.asyncio
async def test_state_management():
    """Test state management between nodes."""
    graph = Graph()
    
    # Create nodes that share data
    node1 = DataNode(id="node1", key="shared", value="first")
    node2 = DataNode(id="node2", key="shared", value="second")
    
    graph.add_node(node1)
    graph.add_node(node2)
    
    # Connect nodes
    graph.add_edge("node1", "default", "node2")
    
    # Add entry point
    graph.add_entry_point("main", "node1")
    
    # Run with initial data
    initial_data = {"initial": "value"}
    context = NodeContext(memory={"test": "memory"})
    
    state = await graph.run("main", initial_data, context)
    
    # Check state management
    assert state.data["initial"] == "value"  # Initial data preserved
    assert state.data["shared"] == "second"  # Last value wins
    assert state.context.memory["test"] == "memory"  # Context preserved

@pytest.mark.asyncio
async def test_graph_validation():
    """Test graph validation."""
    graph = Graph()
    
    # Add single node with invalid edge
    node = CounterNode(id="node")
    node.next_nodes["default"] = "nonexistent"
    graph.add_node(node)
    
    # Validate
    errors = graph.validate()
    assert len(errors) == 1
    assert "nonexistent" in errors[0]

@pytest.mark.asyncio
async def test_invalid_entry_point():
    """Test handling of invalid entry points."""
    graph = Graph()
    
    with pytest.raises(ValueError):
        await graph.run("nonexistent") 