"""Base node class for the graph system.

This module defines the Node abstraction for the Graph framework. A Node represents
an individual unit of work (e.g., an LLM call, a tool invocation, a context injection)
that can be executed within a larger workflow. Nodes are validated via Pydantic and
support asynchronous execution for flexible orchestration.

Typical Usage:
    - Create a subclass of Node
    - Override the 'process' method to implement custom logic
    - Use 'next_nodes' to specify transitions to other nodes in the graph
"""

import abc
from typing import Dict, Any, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field, model_validator

@runtime_checkable
class NodeStateProtocol(Protocol):
    """Protocol defining the interface for NodeState."""
    data: Dict[str, Any]
    results: Dict[str, Any]
    errors: Dict[str, str]

class Node(BaseModel, abc.ABC):
    """
    Abstract base class for all nodes in the Graph.

    Attributes:
        id: Unique identifier for the node. Must be set before adding the node to the graph.
        next_nodes: A mapping of transition keys to follow-up node IDs.
        metadata: Arbitrary metadata for node configuration.
        parallel: If True, this node can run in parallel with others in the graph.
        input_map: Optional mapping of parameter names to keys in NodeState data or results.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this node in the graph."
    )
    next_nodes: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of transition keys to next node IDs."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata or configuration for this node."
    )
    parallel: bool = Field(
        default=False,
        description="If True, this node can be executed in parallel with others."
    )
    input_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of parameter names to keys in NodeState data or results."
    )

    @model_validator(mode='after')
    def check_id(self) -> 'Node':
        """
        Ensure 'id' is not empty or whitespace.
        """
        if not self.id.strip():
            raise ValueError("Node id cannot be empty or whitespace.")
        return self

    class Config:
        """Pydantic configuration for Node."""
        arbitrary_types_allowed = True

    def get_next_node(self, key: str = "default") -> Optional[str]:
        """
        Retrieve the ID of the next node based on the provided key.

        Args:
            key: Identifier for the transition path. Defaults to 'default'.

        Returns:
            The ID of the next node if found, otherwise None.
        """
        return self.next_nodes.get(key)

    def validate(self) -> bool:
        """
        Validate node configuration.

        Override in subclasses if additional checks are required.

        Returns:
            True if the node is considered valid.
        """
        return True

    def _get_nested_value(self, data: Dict[str, Any], dotted_key: str) -> Any:
        """
        Retrieve a nested value from a dictionary using a dotted key path.

        Args:
            data: The dictionary to search.
            dotted_key: The dotted key path, e.g., 'user.profile.name'.

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

    def _prepare_input_data(self, state: NodeStateProtocol) -> Dict[str, Any]:
        """
        Prepare input data for the node based on input_map and NodeState.

        Args:
            state: The current NodeState containing data and results.

        Returns:
            A dictionary of input data for the node.

        Raises:
            ValueError: If a mapped key is not found in state data or results.
        """
        input_data = {}
        for param_name, state_key in self.input_map.items():
            try:
                # Try to get the value from state.data
                input_data[param_name] = self._get_nested_value(state.data, state_key)
                continue
            except ValueError:
                pass
            # Try to get the value from state.results
            found = False
            for result in state.results.values():
                if isinstance(result, dict):
                    try:
                        input_data[param_name] = self._get_nested_value(result, state_key)
                        found = True
                        break
                    except ValueError:
                        continue
            if not found:
                raise ValueError(
                    f"Key '{state_key}' not found in state data or results for parameter '{param_name}'."
                )
        return input_data

    @abc.abstractmethod
    async def process(self, state: NodeStateProtocol) -> Optional[str]:
        """
        Asynchronously process the node's logic.

        This method must be implemented by subclasses and can:
          1. Read or update the graph state via the provided 'state' object.
          2. Perform LLM calls, tool executions, or other logic.
          3. Return the identifier of the next node or None if this node is terminal.

        Args:
            state: An instance of NodeState for data and result handling.

        Returns:
            The ID of the next node to execute, or None if there is no subsequent node.
        """
        raise NotImplementedError("Subclasses must implement this method.") 