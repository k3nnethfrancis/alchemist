"""State management for the graph system.

This module provides:
1. NodeStatus: An enumeration of node execution statuses.
2. NodeStateProtocol: A protocol for classes that manage node-related data.
3. NodeState: A concrete implementation of NodeStateProtocol that stores node results,
   shared data, and error information. It uses Pydantic for basic validation and typing.
4. StateManager: Manages persistence and retrieval of NodeState instances in memory.
   - Future plans: Extend this to support checkpointing, e.g., storing partial state
     in Supabase, files, or other external systems for scalability or reliability.

Checkpointing Approach (Planned):
---------------------------------
- The idea is to periodically persist NodeState to an external database or file,
  so if a workflow crashes or needs to resume, we can reconstruct the state.
- Implementation details will depend on your environment (e.g., Supabase, Redis, S3).

Example (Pseudo-Code):
----------------------
>>> manager = StateManager(config={"storage": "supabase"})
>>> current_state = manager.create_state()
>>> # ... run part of the graph ...
>>> manager.persist_state("workflow123_checkpoint1", current_state)
>>> # ... in case of crash, retrieve it:
>>> resumed_state = manager.retrieve_state("workflow123_checkpoint1")

"""

from typing import Dict, Any, Optional, Set, Protocol
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, model_validator

class NodeStatus(str, Enum):
    """
    Node execution status.

    Attributes:
        PENDING: Node is ready but not yet running.
        RUNNING: Node is currently processing.
        COMPLETED: Node finished successfully.
        ERROR: Node encountered an error.
        SKIPPED: Node was skipped (not used in flow).
        TERMINAL: Node is terminal with no next node.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"
    TERMINAL = "terminal"

class NodeStateProtocol(Protocol):
    """Protocol defining the required state interface for nodes."""
    
    data: Dict[str, Any]
    results: Dict[str, Dict[str, Any]]
    errors: Dict[str, str]
    status: Dict[str, NodeStatus]
    parallel_tasks: Set[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def mark_status(self, node_id: str, status: NodeStatus) -> None: ...
    def add_parallel_task(self, node_id: str) -> None: ...
    def remove_parallel_task(self, node_id: str) -> None: ...
    def get_result(self, node_id: str, key: str) -> Any: ...
    def set_result(self, node_id: str, key: str, value: Any) -> None: ...
    def get_data(self, key: str) -> Any: ...
    def set_data(self, key: str, value: Any) -> None: ...
    def add_error(self, node_id: str, error: str) -> None: ...
    def get_nested_data(self, dotted_key: str) -> Any: ...
    def get_nested_result(self, node_id: str, dotted_key: str) -> Any: ...

class NodeState(BaseModel):
    """
    Concrete implementation of NodeStateProtocol.
    
    Attributes:
        data: Global data shared across workflow
        results: Node-specific results
        errors: Error messages by node ID
        status: Node execution status
        parallel_tasks: Currently running parallel tasks
        metadata: Additional workflow metadata
        created_at: Time of state creation
        updated_at: Time of last state modification
    """
    data: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    status: Dict[str, NodeStatus] = Field(default_factory=dict)
    parallel_tasks: Set[str] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def mark_status(self, node_id: str, status: NodeStatus) -> None:
        """Mark a node's execution status."""
        self.status[node_id] = status
        self._update_timestamp()

    def add_parallel_task(self, node_id: str) -> None:
        """Add a node to the set of parallel tasks."""
        self.parallel_tasks.add(node_id)
        self._update_timestamp()

    def remove_parallel_task(self, node_id: str) -> None:
        """Remove a node from the set of parallel tasks."""
        self.parallel_tasks.discard(node_id)
        self._update_timestamp()

    def get_result(self, node_id: str, key: str) -> Any:
        """Get a specific result value for a node."""
        return self.results.get(node_id, {}).get(key)

    def set_result(self, node_id: str, key: str, value: Any) -> None:
        """Set a specific result value for a node."""
        if node_id not in self.results:
            self.results[node_id] = {}
        self.results[node_id][key] = value
        self._update_timestamp()

    def get_data(self, key: str) -> Any:
        """Get a value from shared data."""
        return self.data.get(key)

    def set_data(self, key: str, value: Any) -> None:
        """Set a value in shared data."""
        self.data[key] = value
        self._update_timestamp()

    def add_error(self, node_id: str, error: str) -> None:
        """Add an error message for a node."""
        self.errors[node_id] = error
        self._update_timestamp()

    def get_nested_data(self, dotted_key: str) -> Any:
        """
        Retrieve a nested value from state data using a dotted key path.

        Args:
            dotted_key: The dotted key path.

        Returns:
            The value found at the specified key path.

        Raises:
            ValueError: If the key path does not exist in state data.
        """
        return self._get_nested_value(self.data, dotted_key)

    def get_nested_result(self, node_id: str, dotted_key: str) -> Any:
        """
        Retrieve a nested value from a node's result using a dotted key path.

        Args:
            node_id: The ID of the node whose result to search.
            dotted_key: The dotted key path.

        Returns:
            The value found at the specified key path in the node's results.

        Raises:
            ValueError: If the key path does not exist in the node's results.
        """
        if node_id not in self.results:
            raise ValueError(f"No results found for node '{node_id}'")
        return self._get_nested_value(self.results[node_id], dotted_key)

    @classmethod
    def _get_nested_value(cls, data: Dict[str, Any], dotted_key: str) -> Any:
        """Get a nested dictionary value using a dotted key path."""
        parts = dotted_key.split('.')
        current = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Key '{part}' not found while traversing '{dotted_key}'")
            current = current[part]
        return current

    def _update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        object.__setattr__(self, "updated_at", datetime.utcnow())

class StateManager(BaseModel):
    """
    Manages persistence and retrieval of NodeState instances.
    
    Attributes:
        config: Configuration for state management
        states: In-memory storage of states
    """
    config: Dict[str, Any] = Field(default_factory=dict)
    states: Dict[str, NodeState] = Field(default_factory=dict)

    def create_state(self) -> NodeState:
        """Create a new NodeState instance."""
        return NodeState()

    def persist_state(self, key: str, state: NodeState) -> None:
        """Store a state instance."""
        self.states[key] = state

    def retrieve_state(self, key: str) -> Optional[NodeState]:
        """Retrieve a stored state."""
        return self.states.get(key)

    def clear_state(self, key: str) -> None:
        """Remove a persisted NodeState by key."""
        self.states.pop(key, None) 