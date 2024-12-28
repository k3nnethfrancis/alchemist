"""Graph Base Classes

This module defines the core graph system:
1. NodeState - Manages state between nodes
2. Graph - Manages node connections and execution
"""

from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class NodeContext(BaseModel):
    """
    Context data shared between nodes.
    
    Attributes:
        memory: Persistent memory between executions
        metadata: Additional metadata for node processing
    """
    
    memory: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NodeState(BaseModel):
    """
    State passed between nodes during execution.
    
    Attributes:
        context: Shared context data
        results: Results from node execution
        data: Temporary data for current execution
    """
    
    context: NodeContext = Field(default_factory=NodeContext)
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)

class Node(BaseModel):
    """
    Base class for all graph nodes.
    
    Attributes:
        id: Unique node identifier
        next_nodes: Mapping of output keys to next node ids
    """
    
    id: str
    next_nodes: Dict[str, Optional[str]] = Field(default_factory=dict)
    
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
    - Graph execution
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
        
    def add_edge(self, from_id: str, key: str, to_id: str):
        """Add edge between nodes."""
        if from_id not in self.nodes:
            raise ValueError(f"Source node {from_id} not found")
        if to_id not in self.nodes:
            raise ValueError(f"Target node {to_id} not found")
            
        self.nodes[from_id].next_nodes[key] = to_id
        if self.debug:
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
    
    async def run(
        self,
        entry_point: str,
        state: Optional[NodeState] = None,
    ) -> NodeState:
        """Run graph from entry point."""
        if entry_point not in self.entry_points:
            raise ValueError(f"Invalid entry point: {entry_point}")
        
        # Use provided state or create new one
        current_state = state or NodeState()
        
        # Start execution
        current_id = self.entry_points[entry_point]
        
        if self.debug:
            logger.info(f"\nStarting graph execution from: {current_id}")
        
        while current_id:
            node = self.nodes[current_id]
            try:
                if self.debug:
                    logger.info(f"\nExecuting node: {current_id}")
                    
                current_id = await node.process(current_state)
                
                if self.debug:
                    logger.info(f"Node {node.id} results: {current_state.results.get(node.id, {})}")
                    if current_id:
                        logger.info(f"Next node: {current_id}")
                    else:
                        logger.info("Execution complete")
                        
            except Exception as e:
                current_state.results[node.id] = {"error": str(e)}
                if self.debug:
                    logger.error(f"Error in node {current_id}: {str(e)}")
                current_id = node.next_nodes.get("error")
                
        return current_state 