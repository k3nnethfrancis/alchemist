"""Graph Base Classes

This module defines the core graph system:
1. Graph - Manages node connections, execution, and state orchestration.
2. test_graph() - A simple test function demonstrating basic graph usage.
"""

from typing import Dict, Any, Optional, Union, List, Set
import logging
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from alchemist.ai.base.logging import (
    AlchemistLoggingConfig, 
    log_verbose, 
    VerbosityLevel, 
    LogComponent
)
from alchemist.ai.graph.state import NodeState, StateManager, NodeStatus
from alchemist.ai.graph.config import GraphConfig
from alchemist.ai.graph.nodes.base.node import Node

# Get logger for graph component
logger = logging.getLogger(LogComponent.GRAPH.value)

class Graph(BaseModel):
    """
    Core graph system for composing agent workflows.

    Attributes:
        nodes: Dictionary mapping node IDs to Node instances
        entry_points: Dictionary mapping entry point names to starting node IDs
        subgraphs: Dictionary of nested Graph instances
        config: Graph-level configuration
        state_manager: Manages state persistence and retrieval
        logging_config: Controls verbosity and logging behavior
    """
    nodes: Dict[str, Node] = Field(default_factory=dict)
    entry_points: Dict[str, str] = Field(default_factory=dict)
    subgraphs: Dict[str, "Graph"] = Field(default_factory=dict)
    config: GraphConfig = Field(default_factory=GraphConfig)
    state_manager: StateManager = Field(default_factory=StateManager)
    logging_config: AlchemistLoggingConfig = Field(
        default_factory=AlchemistLoggingConfig,
        description="Controls verbosity for graph and node transitions"
    )

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Configure logging after model initialization."""
        logger.setLevel(self.logging_config.level)

    def add_node(self, node: Node) -> None:
        """
        Register a node with the graph.

        Args:
            node: Node instance to add.

        Raises:
            ValueError: If the node has no ID or fails validation.
        """
        if not node.id:
            raise ValueError("Node must have an id set")

        if not node.validate():
            raise ValueError(f"Node {node.id} failed validation")

        self._configure_node(node)
        self.nodes[node.id] = node
        logger.info(f"Added node: {node.id} of type {type(node).__name__}")

    def _configure_node(self, node: Node) -> None:
        """
        Configure a node based on graph or node-specific configuration.

        Args:
            node: The node to configure.
        """
        node_config = self.config.get_node_config(type(node).__name__)
        if node_config:
            node.metadata.update(node_config)

    def add_entry_point(self, name: str, node_id: str) -> None:
        """
        Define a named entry point for starting graph execution.

        Args:
            name: Unique name for the entry point.
            node_id: The ID of the node at which execution should begin.

        Raises:
            ValueError: If there is no node with `node_id`.
        """
        if node_id not in self.nodes:
            raise ValueError(f"No node found with id '{node_id}'")
        self.entry_points[name] = node_id
        logger.info(f"Added entry point '{name}' at node: {node_id}")

    def set_entry_point(self, node_id: str) -> None:
        """
        Convenience method to designate a single default entry point.

        Args:
            node_id: The ID of the node which will serve as the graph's primary entry point.
        """
        if node_id not in self.nodes:
            raise ValueError(f"No node found with id '{node_id}'")
        # Use 'default' or any standard name to hold the single entry point
        self.entry_points["default"] = node_id
        logger.info(f"Set default entry point to node: {node_id}")

    def compose(self, other: "Graph", entry_point: str, namespace: Optional[str] = None) -> None:
        """
        Incorporate another graph as a subgraph.

        Args:
            other: The other graph to incorporate.
            entry_point: The name of the entry point in `other` graph to link from.
            namespace: An optional namespace to prefix node IDs, allowing avoidance of collisions.

        Raises:
            ValueError: If the entry_point is not found in the other graph or ID collisions occur.
        """
        if entry_point not in other.entry_points:
            raise ValueError(f"Entry point '{entry_point}' not found in subgraph")

        prefix = f"{namespace}." if namespace else ""

        for node_id, node in other.nodes.items():
            new_id = f"{prefix}{node_id}"
            if new_id in self.nodes:
                raise ValueError(f"Node ID collision: {new_id}")

            node_copy = node.copy()
            node_copy.id = new_id

            # Update next_nodes references to the new namespace
            for key, next_id in node.next_nodes.items():
                if next_id:
                    node_copy.next_nodes[key] = f"{prefix}{next_id}"

            self.nodes[new_id] = node_copy

        if namespace:
            self.subgraphs[namespace] = other

        logger.info(
            f"Composed subgraph with entry point '{entry_point}' "
            f"under namespace '{namespace or 'no-namespace'}'"
        )

    def validate(self) -> List[str]:
        """
        Validate the entire graph configuration, including node references and entry points.

        Returns:
            A list of validation error messages. Empty if the graph is valid.
        """
        errors: List[str] = []

        for node_id, node in self.nodes.items():
            if not node.validate():
                errors.append(f"Node {node_id} failed validation")

            for key, next_id in node.next_nodes.items():
                if next_id and next_id not in self.nodes:
                    errors.append(f"Node {node_id} references unknown node: {next_id}")

        for name, node_id in self.entry_points.items():
            if node_id not in self.nodes:
                errors.append(f"Entry point '{name}' references unknown node: {node_id}")

        return errors

    async def run(
        self,
        entry_point: Optional[str] = None,
        state: Optional[NodeState] = None,
        state_key: Optional[str] = None
    ) -> NodeState:
        """
        Run the graph from a specified entry point.

        This method:
            1. Resolves the NodeState (creates or retrieves).
            2. Validates the entry point node.
            3. Iteratively processes each node until a terminal is reached or no next node.

        Logging Details:
            - At VERBOSE or DEBUG levels, logs transitions between nodes.
            - At INFO level, logs only major steps.

        Args:
            entry_point: Name of the entry point (key in self.entry_points).
            state: Optional existing NodeState.
            state_key: Optional key for retrieving a stored state.

        Returns:
            The final NodeState after execution.
        """
        if state is None:
            state = NodeState()

        # Use "default" if no entry_point specified
        entry_point = entry_point or "default"
        
        if entry_point not in self.entry_points:
            raise ValueError(f"No entry point found: {entry_point}")
        
        start_node_id = self.entry_points[entry_point]

        if self.logging_config.show_node_transitions or \
           self.logging_config.level <= VerbosityLevel.DEBUG:
            log_verbose(logger, f"Starting graph at node '{start_node_id}'")

        try:
            current_node_id: Optional[str] = start_node_id
            parallel_tasks: Set[asyncio.Task] = set()

            while current_node_id or parallel_tasks:
                if current_node_id:
                    node = self.nodes[current_node_id]

                    if node.parallel:
                        # Run this node in parallel
                        task = asyncio.create_task(self._process_node(node, state))
                        parallel_tasks.add(task)
                        current_node_id = node.get_next_node()
                        continue

                    # Sequential node processing
                    current_node_id = await self._process_node(node, state)

                # If any parallel tasks exist, wait for at least one to complete
                if parallel_tasks:
                    done, parallel_tasks = await asyncio.wait(
                        parallel_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        # Raise any exception from the parallel task
                        await task

            if state_key:
                self.state_manager.persist_state(state_key, state)

            return state

        except Exception as e:
            logger.error(f"Error executing graph: {str(e)}", exc_info=True)
            state.add_error("graph", str(e))
            raise

    async def _process_node(self, node: Node, state: NodeState) -> Optional[str]:
        """
        Process a single node, handle any errors, and return the ID of the next node (if any).

        At VERBOSE or DEBUG levels, logs node status changes.

        Args:
            node: The node to process.
            state: The current NodeState.

        Returns:
            The ID of the next node or None if the flow ends.
        """
        # Log node start
        state.mark_status(node.id, NodeStatus.RUNNING)
        if self.logging_config.show_node_transitions or \
           self.logging_config.level <= VerbosityLevel.DEBUG:
            log_verbose(logger, f"Node '{node.id}' status: RUNNING")

        try:
            logger.debug(f"Processing node: {node.id}")
            next_node_id = await node.process(state)

            if state.status[node.id] != NodeStatus.TERMINAL:
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
        """
        Retrieve or create a state, optionally loading from StateManager if a key is given.

        Args:
            state: An existing NodeState instance (if any).
            state_key: State key for persistence. If provided and matches a stored state,
                       that stored state is returned.

        Returns:
            A valid NodeState object to use for execution.
        """
        if state_key:
            loaded_state = self.state_manager.retrieve_state(state_key)
            if loaded_state:
                return loaded_state
        return state or self.state_manager.create_state()

    def add_edge(self, from_node_id: str, transition_key: str, to_node_id: str) -> None:
        """
        A convenience method to connect two nodes in the graph.

        Args:
            from_node_id: The ID of the node that will transition to another node.
            transition_key: The key on the node's next_nodes dictionary.
            to_node_id: The node ID that follows.
        """
        if from_node_id not in self.nodes:
            raise ValueError(f"No node found with id '{from_node_id}'")
        if to_node_id not in self.nodes:
            raise ValueError(f"No node found with id '{to_node_id}'")

        self.nodes[from_node_id].next_nodes[transition_key] = to_node_id

    def chain(self, nodes: List[Node]) -> None:
        """
        Chain nodes in sequence, setting the next_node of each to the following node.

        Args:
            nodes: A list of Node instances to chain together.

        Raises:
            ValueError: If the list of nodes is empty.
        """
        if not nodes:
            raise ValueError("Cannot chain an empty list of nodes.")

        for i, node in enumerate(nodes):
            self.add_node(node)
            if i < len(nodes) - 1:
                node.next_nodes["default"] = nodes[i + 1].id

    async def run_loop(self, entry_point: str, state: Optional[NodeState] = None,
                       state_key: Optional[str] = None, max_loops: int = 1) -> NodeState:
        """
        Run the graph starting from the specified entry point, with support for looping.

        Args:
            entry_point: The name of the entry point to start execution from.
            state: An optional NodeState instance.
            state_key: Optional key for state persistence.
            max_loops: Maximum number of times to loop the execution.

        Returns:
            The final NodeState after execution.

        Raises:
            ValueError: If entry point is invalid or max_loops is non-positive.
        """
        if max_loops <= 0:
            raise ValueError("max_loops must be a positive integer.")

        for _ in range(max_loops):
            state = await self.run(entry_point, state, state_key)
            # Add your condition to break the loop if needed
            # For now, it will loop max_loops times
        return state

async def test_graph() -> None:
    """
    Test the graph framework functionality with sample local nodes.

    This function:
    1. Defines simple nodes that perform arithmetic.
    2. Demonstrates serial and parallel node execution.
    3. Shows how to attach context suppliers to NodeState.
    """
    import asyncio

    # Local test Node definitions (example only)
    class AddNode(Node):
        async def process(self, state: NodeState) -> Optional[str]:
            a = state.get_data("a") or 0
            b = state.get_data("b") or 0
            state.set_result(self.id, "sum", a + b)
            return self.get_next_node()

    class MultiplyNode(Node):
        async def process(self, state: NodeState) -> Optional[str]:
            a = state.get_data("a") or 1
            b = state.get_data("b") or 1
            state.set_result(self.id, "product", a * b)
            return self.get_next_node()

    class SlowNode(Node):
        parallel = True

        async def process(self, state: NodeState) -> Optional[str]:
            await asyncio.sleep(1)
            state.set_result(self.id, "done", True)
            return self.get_next_node()

    # Build a sample graph
    graph = Graph()

    add_node = AddNode(id="add")
    mult_node = MultiplyNode(id="multiply")
    slow_node1 = SlowNode(id="slow1")
    slow_node2 = SlowNode(id="slow2")

    graph.add_node(add_node)
    graph.add_node(mult_node)
    graph.add_node(slow_node1)
    graph.add_node(slow_node2)

    graph.add_edge("add", "default", "multiply")
    graph.add_edge("multiply", "default", "slow1")
    graph.add_edge("slow1", "default", "slow2")

    graph.add_entry_point("main", "add")

    # Validate and run
    errors = graph.validate()
    assert not errors, f"Graph validation failed: {errors}"

    from datetime import datetime

    state = NodeState()
    state.set_data("a", 5)
    state.set_data("b", 3)

    # Example of adding a context supplier
    async def time_supplier(**kwargs) -> str:
        return datetime.now().isoformat()

    # This code references 'context' usage; adapt if you
    # have a separate context mechanism:
    # state.context.add_supplier("time", time_supplier)

    start_time = datetime.now()
    final_state = await graph.run("main", state)
    end_time = datetime.now()

    # Verify results
    assert final_state.results["add"]["sum"] == 8
    assert final_state.results["multiply"]["product"] == 15
    assert final_state.results["slow1"]["done"] is True
    assert final_state.results["slow2"]["done"] is True

    # Parallel execution check
    duration = (end_time - start_time).total_seconds()
    assert duration < 3.0, "Parallel execution took too long."

    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_graph()) 