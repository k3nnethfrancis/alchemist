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

from typing import Dict, Any, Optional, Set
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


class NodeStateProtocol:
    """
    Protocol defining the required state interface for nodes.
    """

    def mark_status(self, node_id: str, status: NodeStatus) -> None:
        ...

    def add_parallel_task(self, node_id: str) -> None:
        ...

    def remove_parallel_task(self, node_id: str) -> None:
        ...

    def get_result(self, node_id: str, key: str) -> Any:
        ...

    def set_result(self, node_id: str, key: str, value: Any) -> None:
        ...

    def get_data(self, key: str) -> Any:
        ...

    def set_data(self, key: str, value: Any) -> None:
        ...

    def add_error(self, node_id: str, error: str) -> None:
        ...

    @property
    def results(self) -> Dict[str, Dict[str, Any]]:
        ...

    @property
    def data(self) -> Dict[str, Any]:
        ...

    @property
    def errors(self) -> Dict[str, str]:
        ...

    @property
    def status(self) -> Dict[str, NodeStatus]:
        ...

    @property
    def parallel_tasks(self) -> Set[str]:
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        ...

    @property
    def created_at(self) -> datetime:
        ...

    @property
    def updated_at(self) -> datetime:
        ...


class NodeState(BaseModel):
    """
    Concrete implementation of NodeStateProtocol using Pydantic.

    Attributes:
        results: Dict of node results, keyed by node ID.
        data: Shared input data or context.
        errors: Recorded errors or exceptions.
        status: Execution status of nodes.
        created_at: Timestamp when the state was created.
        updated_at: Timestamp when the state was last updated.
    """

    results: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    status: Dict[str, NodeStatus] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def _get_nested_value(cls, data: Dict[str, Any], dotted_key: str) -> Any:
        """
        Retrieve a nested value from data or results using a dotted key path.

        Args:
            data: The data dictionary to search.
            dotted_key: The dotted key path.

        Returns:
            The value found at the specified key path.

        Raises:
            ValueError: If the key path does not exist in the data.
        """
        parts = dotted_key.split('.')
        current = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Key '{part}' not found while traversing '{dotted_key}'")
            current = current[part]
        return current

    @model_validator(mode='after')
    def update_timestamp(self) -> 'NodeState':
        self.updated_at = datetime.utcnow()
        return self

    def mark_status(self, node_id: str, status: NodeStatus) -> None:
        self.status[node_id] = status

    def add_parallel_task(self, node_id: str) -> None:
        self.mark_status(node_id, NodeStatus.RUNNING)

    def remove_parallel_task(self, node_id: str) -> None:
        if node_id in self.status:
            self.status[node_id] = NodeStatus.COMPLETED

    def get_result(self, node_id: str, key: str) -> Any:
        return self.results.get(node_id, {}).get(key)

    def set_result(self, node_id: str, key: str, value: Any) -> None:
        if node_id not in self.results:
            self.results[node_id] = {}
        self.results[node_id][key] = value

    def get_data(self, key: str) -> Any:
        return self.data.get(key)

    def set_data(self, key: str, value: Any) -> None:
        self.data[key] = value

    def add_error(self, node_id: str, error: str) -> None:
        self.errors[node_id] = error

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


class StateManager:
    """
    Manages state persistence and retrieval for NodeState objects.

    By default, it stores states in an in-memory dictionary. For large or distributed
    workflows, you can extend or override to connect to external systems.

    Potential Checkpointing Strategies:
        - Store NodeState in a Postgres or Supabase table
        - Serialize to JSON and save in object storage
        - Write to Redis for ephemeral caching

    Attributes:
        config: Optional configuration for the manager.
        _states: In-memory dictionary of NodeStates keyed by string IDs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the StateManager.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self._states: Dict[str, NodeState] = {}

    def create_state(self) -> NodeState:
        """
        Create a new, empty NodeState.

        Returns:
            A newly created NodeState object.
        """
        return NodeState()

    def persist_state(self, key: str, state: NodeState) -> None:
        """
        Persist a NodeState under a given key. By default, only in memory.

        Future expansions might store states in an external system or a file.
        """
        self._states[key] = state

    def retrieve_state(self, key: str) -> Optional[NodeState]:
        """
        Retrieve a previously persisted NodeState by key.

        Returns:
            The NodeState if found, or None otherwise.
        """
        return self._states.get(key)

    def clear_state(self, key: str) -> None:
        """
        Remove a persisted NodeState by key, if it exists.
        """
        self._states.pop(key, None) 