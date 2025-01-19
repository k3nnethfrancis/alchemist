"""Graph Base Classes

This module defines the core graph system:
1. NodeContext - Manages shared context and memory between nodes
2. NodeState - Manages state between nodes
3. Graph - Manages node connections and execution
"""

from typing import Dict, Any, Optional, Union, List, Type, Set
from pydantic import BaseModel, Field
import logging
import asyncio
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)

class NodeStatus(str, Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"
    TERMINAL = "terminal"  # New status for terminal nodes

class NodeContext(BaseModel):
    """
    Context data shared between nodes.
    
    Attributes:
        memory: Persistent memory between executions
        metadata: Additional metadata for node processing
        checkpoints: Dictionary of saved states for recovery
        suppliers: Dictionary of context supplier functions
    """
    
    memory: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checkpoints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    suppliers: Dict[str, Any] = Field(default_factory=dict)

    def save_checkpoint(self, name: str, state: Dict[str, Any]):
        """Save a named checkpoint of the current state."""
        self.checkpoints[name] = {
            "state": state,
            "timestamp": datetime.now().isoformat()
        }
    
    def load_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a named checkpoint."""
        if name in self.checkpoints:
            return self.checkpoints[name]["state"]
        return None

    def add_supplier(self, name: str, supplier: Any):
        """Add a context supplier function."""
        self.suppliers[name] = supplier

    async def get_context(self, name: str, **kwargs) -> Any:
        """Get context from a named supplier."""
        if name not in self.suppliers:
            raise ValueError(f"Context supplier {name} not found")
        return await self.suppliers[name](**kwargs)

class NodeState(BaseModel):
    """
    State passed between nodes during execution.
    
    Attributes:
        context: Shared context data
        results: Results from node execution
        data: Temporary data for current execution
        status: Status of each node's execution
        parallel_tasks: Set of nodes running in parallel
    """
    
    context: NodeContext = Field(default_factory=NodeContext)
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    status: Dict[str, NodeStatus] = Field(default_factory=dict)
    parallel_tasks: Set[str] = Field(default_factory=set)

    def mark_status(self, node_id: str, status: NodeStatus):
        """Update node execution status."""
        self.status[node_id] = status

    def add_parallel_task(self, node_id: str):
        """Add node to parallel execution set."""
        self.parallel_tasks.add(node_id)

    def remove_parallel_task(self, node_id: str):
        """Remove node from parallel execution set."""
        self.parallel_tasks.remove(node_id)

class Node(BaseModel):
    """
    Base class for all graph nodes.
    
    Attributes:
        id: Unique node identifier
        next_nodes: Mapping of output keys to next node ids
        parallel: Whether this node can run in parallel
    """
    
    id: str
    next_nodes: Dict[str, Optional[str]] = Field(default_factory=dict)
    parallel: bool = False
    
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Process the current state and return next node id.
        
        Args:
            state: Current node state
            
        Returns:
            ID of next node to execute
        """
        raise NotImplementedError

class Graph:
    """
    Manages a collection of connected nodes.
    
    Features:
    - Node addition and removal
    - Edge validation
    - Graph execution with parallel support
    - State checkpointing
    - Context suppliers
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.entry_points: Dict[str, str] = {}
        self.debug = True  # Enable debug output
    
    def add_node(self, node: Node):
        """Add node to graph."""
        self.nodes[node.id] = node
        if self.debug:
            logger.info(f"Added node: {node.id}")
        
    def add_edge(self, from_id: str, key: str, to_id: Optional[str]):
        """Add edge between nodes.
        
        Args:
            from_id: Source node ID
            key: Edge key/label
            to_id: Target node ID or None for terminal nodes
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node {from_id} not found")
        if to_id is not None and to_id not in self.nodes:
            raise ValueError(f"Target node {to_id} not found")
            
        self.nodes[from_id].next_nodes[key] = to_id
        if self.debug:
            if to_id is None:
                logger.info(f"Added terminal edge: {from_id} --[{key}]--> [END]")
            else:
                logger.info(f"Added edge: {from_id} --[{key}]--> {to_id}")
        
    def add_entry_point(self, name: str, node_id: str):
        """Add named entry point to graph."""
        if node_id not in self.nodes:
            raise ValueError(f"Entry point node {node_id} not found")
            
        self.entry_points[name] = node_id
        if self.debug:
            logger.info(f"Added entry point '{name}' at node: {node_id}")
    
    def validate(self) -> List[str]:
        """
        Validate graph connections.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check all next_nodes point to valid nodes
        for node in self.nodes.values():
            for key, next_id in node.next_nodes.items():
                if next_id is not None and next_id not in self.nodes:
                    errors.append(
                        f"Node {node.id} has invalid edge '{key}' to non-existent node {next_id}"
                    )
        
        if self.debug:
            if errors:
                logger.warning("Validation errors found:")
                for error in errors:
                    logger.warning(f"  - {error}")
            else:
                logger.info("Graph validation successful")
                    
        return errors
    
    def _log_state(self, node_id: str, state: NodeState, phase: str):
        """Log detailed state information.
        
        Args:
            node_id: ID of current node
            state: Current state
            phase: Execution phase (pre/post)
        """
        if not self.debug:
            return
            
        logger.info(f"\n{'='*50}")
        logger.info(f"Node {node_id} - {phase} Processing")
        logger.info(f"{'='*50}")
        
        # Log node status
        logger.info("\nNode Status:")
        for nid, status in state.status.items():
            logger.info(f"  {nid}: {status}")
        
        # Log results (excluding large data)
        logger.info("\nResults:")
        for nid, result in state.results.items():
            if nid == "content":  # Skip large content
                continue
            try:
                result_str = json.dumps(result, indent=2)
                logger.info(f"\n  {nid}:")
                for line in result_str.splitlines():
                    logger.info(f"    {line}")
            except Exception as e:
                logger.info(f"  {nid}: <error serializing: {str(e)}>")
        
        # Log parallel tasks
        if state.parallel_tasks:
            logger.info("\nParallel Tasks:")
            for task in state.parallel_tasks:
                logger.info(f"  - {task}")
        
        logger.info(f"\n{'='*50}\n")
    
    async def run(
        self,
        entry_point: str,
        state: Optional[NodeState] = None,
    ) -> NodeState:
        """Run graph from entry point with parallel execution support."""
        if entry_point not in self.entry_points:
            raise ValueError(f"Invalid entry point: {entry_point}")
        
        # Use provided state or create new one
        current_state = state or NodeState()
        
        # Start execution
        current_id = self.entry_points[entry_point]
        
        if self.debug:
            logger.info(f"\nStarting graph execution from: {current_id}")
            self._log_state(current_id, current_state, "Initial")
        
        while current_id or current_state.parallel_tasks:
            if current_id:
                node = self.nodes[current_id]
                current_state.mark_status(node.id, NodeStatus.RUNNING)
                
                try:
                    if node.parallel:
                        # Start parallel execution
                        current_state.add_parallel_task(node.id)
                        asyncio.create_task(self._run_parallel_node(node, current_state))
                        current_id = None  # Don't wait for result
                    else:
                        if self.debug:
                            logger.info(f"\nExecuting node: {current_id}")
                            self._log_state(current_id, current_state, "Pre")
                            
                        next_id = await node.process(current_state)
                        
                        if next_id is None:
                            current_state.mark_status(node.id, NodeStatus.TERMINAL)
                            if self.debug:
                                logger.info(f"Node {node.id} reached terminal state")
                                self._log_state(node.id, current_state, "Terminal")
                        else:
                            current_state.mark_status(node.id, NodeStatus.COMPLETED)
                            if self.debug:
                                logger.info(f"Node {node.id} completed, next: {next_id}")
                                self._log_state(node.id, current_state, "Post")
                        
                        current_id = next_id
                            
                except Exception as e:
                    current_state.mark_status(node.id, NodeStatus.ERROR)
                    current_state.results[node.id] = {"error": str(e)}
                    if self.debug:
                        logger.error(f"Error in node {current_id}: {str(e)}")
                        self._log_state(current_id, current_state, "Error")
                    current_id = node.next_nodes.get("error")
            
            # Wait for parallel tasks if no current node
            if not current_id and current_state.parallel_tasks:
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
        if self.debug:
            logger.info("\nExecution complete")
            self._log_state("final", current_state, "Complete")
            
        return current_state
    
    async def _run_parallel_node(self, node: Node, state: NodeState):
        """Execute a node in parallel."""
        try:
            if self.debug:
                self._log_state(node.id, state, "Pre-Parallel")
                
            next_id = await node.process(state)
            
            if next_id is None:
                state.mark_status(node.id, NodeStatus.TERMINAL)
                if self.debug:
                    self._log_state(node.id, state, "Terminal-Parallel")
            else:
                state.mark_status(node.id, NodeStatus.COMPLETED)
                if self.debug:
                    self._log_state(node.id, state, "Post-Parallel")
                    
        except Exception as e:
            state.mark_status(node.id, NodeStatus.ERROR)
            state.results[node.id] = {"error": str(e)}
            if self.debug:
                logger.error(f"Error in parallel node {node.id}: {str(e)}")
                self._log_state(node.id, state, "Error-Parallel")
        finally:
            state.remove_parallel_task(node.id)

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