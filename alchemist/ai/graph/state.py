"""State management for the graph system."""

from typing import Dict, Any, Optional, Protocol, runtime_checkable
from datetime import datetime
from enum import Enum

class NodeStatus(str, Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"
    TERMINAL = "terminal"

@runtime_checkable
class NodeStateProtocol(Protocol):
    """Protocol defining the required state interface for nodes."""
    def mark_status(self, node_id: str, status: NodeStatus) -> None: ...
    def add_parallel_task(self, node_id: str) -> None: ...
    def remove_parallel_task(self, node_id: str) -> None: ...
    def get_result(self, node_id: str, key: str) -> Any: ...
    def set_result(self, node_id: str, key: str, value: Any) -> None: ...
    def get_data(self, key: str) -> Any: ...
    def set_data(self, key: str, value: Any) -> None: ...
    def add_error(self, node_id: str, error: str) -> None: ...
    
    @property
    def results(self) -> Dict[str, Dict[str, Any]]: ...
    @property
    def data(self) -> Dict[str, Any]: ...
    @property
    def errors(self) -> Dict[str, str]: ...

class NodeState(NodeStateProtocol):
    """
    Concrete implementation of node state.
    
    Attributes:
        results: Results from node executions
        data: Shared data between nodes
        errors: Error messages by node
        status: Node execution status
        parallel_tasks: Set of currently running parallel tasks
        metadata: Additional state metadata
        created_at: State creation timestamp
        updated_at: Last update timestamp
    """
    
    def __init__(self):
        self._results: Dict[str, Dict[str, Any]] = {}
        self._data: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        self._status: Dict[str, NodeStatus] = {}
        self._parallel_tasks: set[str] = set()
        self._metadata: Dict[str, Any] = {}
        self._created_at = datetime.now()
        self._updated_at = self._created_at

    def mark_status(self, node_id: str, status: NodeStatus) -> None:
        """Update node status."""
        self._status[node_id] = status
        self._updated_at = datetime.now()

    def add_parallel_task(self, node_id: str) -> None:
        """Track a parallel task."""
        self._parallel_tasks.add(node_id)
        
    def remove_parallel_task(self, node_id: str) -> None:
        """Remove a completed parallel task."""
        self._parallel_tasks.discard(node_id)

    def get_result(self, node_id: str, key: str) -> Any:
        """Get a specific result value."""
        return self._results.get(node_id, {}).get(key)

    def set_result(self, node_id: str, key: str, value: Any) -> None:
        """Set a result value."""
        if node_id not in self._results:
            self._results[node_id] = {}
        self._results[node_id][key] = value
        self._updated_at = datetime.now()

    def get_data(self, key: str) -> Any:
        """Get shared data value."""
        return self._data.get(key)

    def set_data(self, key: str, value: Any) -> None:
        """Set shared data value."""
        self._data[key] = value
        self._updated_at = datetime.now()

    def add_error(self, node_id: str, error: str) -> None:
        """Record an error."""
        self._errors[node_id] = error
        self._updated_at = datetime.now()

    @property
    def results(self) -> Dict[str, Dict[str, Any]]:
        """Get all results."""
        return self._results

    @property
    def data(self) -> Dict[str, Any]:
        """Get shared data."""
        return self._data

    @property
    def errors(self) -> Dict[str, str]:
        """Get all errors."""
        return self._errors

    @property
    def status(self) -> Dict[str, NodeStatus]:
        """Get all node statuses."""
        return self._status

    @property
    def parallel_tasks(self) -> set[str]:
        """Get currently running parallel tasks."""
        return self._parallel_tasks

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get state metadata."""
        return self._metadata

    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at

class StateManager:
    """
    Manages state persistence and retrieval.
    
    This class handles:
    - State creation
    - State persistence
    - State retrieval
    - State validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._states: Dict[str, NodeState] = {}

    def create_state(self) -> NodeState:
        """Create a new state instance."""
        return NodeState()

    def persist_state(self, key: str, state: NodeState) -> None:
        """
        Persist a state instance.
        
        Args:
            key: Unique state identifier
            state: State to persist
        """
        self._states[key] = state

    def retrieve_state(self, key: str) -> Optional[NodeState]:
        """
        Retrieve a persisted state.
        
        Args:
            key: State identifier
            
        Returns:
            Optional[NodeState]: Retrieved state or None if not found
        """
        return self._states.get(key)

    def clear_state(self, key: str) -> None:
        """
        Remove a persisted state.
        
        Args:
            key: State identifier to remove
        """
        self._states.pop(key, None) 