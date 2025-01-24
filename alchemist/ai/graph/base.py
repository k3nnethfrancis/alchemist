"""Graph Base Classes

This module defines the core graph system:
1. Graph - Manages node connections and execution
"""

from typing import Dict, Any, Optional, Union, List
import logging
import asyncio

from alchemist.ai.base.logging import LogComponent
from alchemist.ai.graph.state import NodeState, StateManager, NodeStatus
from alchemist.ai.graph.config import GraphConfig
from alchemist.ai.graph.nodes.base import Node

# Get logger for graph component
logger = logging.getLogger(LogComponent.GRAPH.value)

class Graph:
    """
    Core graph system for composing agent workflows.

    Features:
    - Node management and validation
    - State orchestration
    - Graph composition
    - Parallel execution
    - Subgraph management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.nodes: Dict[str, Node] = {}
        self.entry_points: Dict[str, str] = {}
        self.subgraphs: Dict[str, 'Graph'] = {}  # Track composed graphs
        self.config = GraphConfig.from_dict(config)
        self.state_manager = StateManager(config)

    def add_node(self, node: Node) -> None:
        """
        Register a node with validation and auto-configuration.
        
        Args:
            node: Node instance to add
            
        Raises:
            ValueError: If node validation fails
        """
        if not node.id:
            raise ValueError("Node must have an id set")
            
        # Validate node
        if not node.validate():
            raise ValueError(f"Node {node.id} failed validation")
            
        # Configure node based on type if needed
        self._configure_node(node)
        
        # Add to graph
        self.nodes[node.id] = node
        logger.info(f"Added node: {node.id} of type {type(node).__name__}")

    def _configure_node(self, node: Node) -> None:
        """Configure node based on its type and graph config."""
        # Apply any type-specific configuration
        node_config = self.config.get_node_config(type(node).__name__)
        if node_config:
            node.metadata.update(node_config)

    def add_entry_point(self, name: str, node_id: str) -> None:
        """
        Define a named entry point for starting execution.
        
        Args:
            name: Entry point name
            node_id: Starting node ID
        """
        if node_id not in self.nodes:
            raise ValueError(f"No node found with id '{node_id}'")
        self.entry_points[name] = node_id
        logger.info(f"Added entry point '{name}' at node: {node_id}")

    def compose(self, other: 'Graph', entry_point: str, namespace: Optional[str] = None) -> None:
        """
        Incorporate another graph as a subgraph.
        
        Args:
            other: Graph to compose with
            entry_point: Entry point name in other graph
            namespace: Optional namespace to prefix node IDs
        """
        if entry_point not in other.entry_points:
            raise ValueError(f"Entry point '{entry_point}' not found in subgraph")
            
        # Create namespace for node IDs if provided
        prefix = f"{namespace}." if namespace else ""
        
        # Copy nodes with namespaced IDs
        for node_id, node in other.nodes.items():
            new_id = f"{prefix}{node_id}"
            if new_id in self.nodes:
                raise ValueError(f"Node ID collision: {new_id}")
                
            # Create copy of node with new ID
            node_copy = node.copy()
            node_copy.id = new_id
            
            # Update next_nodes with new namespaced IDs
            for key, next_id in node.next_nodes.items():
                if next_id:
                    node_copy.next_nodes[key] = f"{prefix}{next_id}"
                    
            self.nodes[new_id] = node_copy
            
        # Store reference to subgraph
        if namespace:
            self.subgraphs[namespace] = other
            
        logger.info(f"Composed subgraph with entry point '{entry_point}' and namespace '{namespace}'")

    async def run(
        self, 
        entry_point: str, 
        state: Optional[NodeState] = None,
        state_key: Optional[str] = None
    ) -> NodeState:
        """
        Execute the graph from an entry point.
        
        Args:
            entry_point: Name of entry point to start from
            state: Optional initial state
            state_key: Optional key for state persistence
            
        Returns:
            Final NodeState after execution
        """
        if entry_point not in self.entry_points:
            raise ValueError(f"No entry point named '{entry_point}'")
            
        # Get or create state
        state = self._get_or_create_state(state, state_key)
        start_node_id = self.entry_points[entry_point]
        
        try:
            # Initialize execution
            current_node_id = start_node_id
            parallel_tasks = set()
            
            while current_node_id or parallel_tasks:
                if current_node_id:
                    node = self.nodes[current_node_id]
                    
                    # Handle parallel nodes
                    if node.parallel:
                        task = asyncio.create_task(self._process_node(node, state))
                        parallel_tasks.add(task)
                        current_node_id = node.get_next_node()
                        continue
                        
                    # Process sequential node
                    current_node_id = await self._process_node(node, state)
                    
                # Wait for any parallel tasks
                if parallel_tasks:
                    done, parallel_tasks = await asyncio.wait(
                        parallel_tasks, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        await task  # Handle any exceptions
                        
            # Persist final state if requested
            if state_key:
                self.state_manager.persist_state(state_key, state)
                
            return state
            
        except Exception as e:
            logger.error(f"Error executing graph: {str(e)}", exc_info=True)
            state.add_error("graph", str(e))
            raise

    async def _process_node(self, node: Node, state: NodeState) -> Optional[str]:
        """Process a single node."""
        try:
            logger.debug(f"Processing node: {node.id}")
            state.mark_status(node.id, NodeStatus.RUNNING)
            
            # Process node
            next_node_id = await node.process(state)
            
            # Update status
            state.mark_status(node.id, NodeStatus.COMPLETED)
            logger.debug(f"Node {node.id} completed, next: {next_node_id}")
            
            return next_node_id
            
        except Exception as e:
            logger.error(f"Error in node {node.id}: {str(e)}", exc_info=True)
            state.mark_status(node.id, NodeStatus.ERROR)
            state.add_error(node.id, str(e))
            return node.get_next_node("error")

    def _get_or_create_state(
        self, 
        state: Optional[NodeState], 
        state_key: Optional[str]
    ) -> NodeState:
        """Get existing state or create new one."""
        if state_key:
            state = self.state_manager.retrieve_state(state_key) or state
        return state or self.state_manager.create_state()

    def validate(self) -> List[str]:
        """
        Validate the entire graph configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate nodes
        for node_id, node in self.nodes.items():
            if not node.validate():
                errors.append(f"Node {node_id} failed validation")
                
            # Check next_nodes references
            for key, next_id in node.next_nodes.items():
                if next_id and next_id not in self.nodes:
                    errors.append(f"Node {node_id} references unknown node {next_id}")
                    
        # Validate entry points
        for name, node_id in self.entry_points.items():
            if node_id not in self.nodes:
                errors.append(f"Entry point {name} references unknown node {node_id}")
                
        return errors

async def test_graph():
    """Test the graph framework functionality."""
    
    # Test node that adds numbers
    class AddNode(Node):
        async def process(self, state: NodeState) -> Optional[str]:
            a = state.data.get("a", 0)
            b = state.data.get("b", 0)
            state.results[self.id] = {"sum": a + b}
            return self.next_nodes.get("default")
    
    # Test node that multiplies numbers
    class MultiplyNode(Node):
        async def process(self, state: NodeState) -> Optional[str]:
            a = state.data.get("a", 1)
            b = state.data.get("b", 1)
            state.results[self.id] = {"product": a * b}
            return self.next_nodes.get("default")
    
    # Test parallel node
    class SlowNode(Node):
        parallel = True
        
        async def process(self, state: NodeState) -> Optional[str]:
            await asyncio.sleep(1)  # Simulate slow operation
            state.results[self.id] = {"done": True}
            return self.next_nodes.get("default")
    
    # Create graph
    graph = Graph()
    
    # Add nodes
    add_node = AddNode(id="add")
    mult_node = MultiplyNode(id="multiply")
    slow_node1 = SlowNode(id="slow1")
    slow_node2 = SlowNode(id="slow2")
    
    graph.add_node(add_node)
    graph.add_node(mult_node)
    graph.add_node(slow_node1)
    graph.add_node(slow_node2)
    
    # Add edges
    graph.add_edge("add", "default", "multiply")
    graph.add_edge("multiply", "default", "slow1")
    graph.add_edge("slow1", "default", "slow2")
    
    # Add entry point
    graph.add_entry_point("main", "add")
    
    # Validate graph
    errors = graph.validate()
    assert not errors, "Graph validation failed"
    
    # Create initial state
    state = NodeState()
    state.data["a"] = 5
    state.data["b"] = 3
    
    # Add a context supplier
    async def time_supplier(**kwargs):
        return datetime.now().isoformat()
    
    state.context.add_supplier("time", time_supplier)
    
    # Run graph
    start_time = datetime.now()
    final_state = await graph.run("main", state)
    end_time = datetime.now()
    
    # Verify results
    assert final_state.results["add"]["sum"] == 8
    assert final_state.results["multiply"]["product"] == 15
    assert final_state.results["slow1"]["done"]
    assert final_state.results["slow2"]["done"]
    
    # Verify parallel execution
    duration = (end_time - start_time).total_seconds()
    assert duration < 3.0, "Parallel execution took too long"
    
    # Test context supplier
    time_str = await final_state.context.get_context("time")
    assert time_str is not None
    
    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_graph()) 