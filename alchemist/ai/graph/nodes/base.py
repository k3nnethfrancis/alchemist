"""Base node class for the graph system."""

import abc
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from alchemist.ai.graph.state import NodeStateProtocol

class Node(BaseModel, abc.ABC):
    """
    Abstract base for all graph nodes.
    
    Attributes:
        id: Unique node identifier
        next_nodes: Mapping of output keys to next node ids
        parallel: Whether this node can run in parallel
        metadata: Additional node configuration
    """
    id: str
    next_nodes: Dict[str, Optional[str]] = Field(default_factory=dict)
    parallel: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @abc.abstractmethod
    async def process(self, state: NodeStateProtocol) -> Optional[str]:
        """
        Process the current state and return next node id.
        
        Args:
            state: Current node state
            
        Returns:
            Optional[str]: ID of next node to execute, or None if terminal
        """
        pass

    def get_next_node(self, key: str = "default") -> Optional[str]:
        """Get next node ID for given transition key."""
        return self.next_nodes.get(key)

    def validate(self) -> bool:
        """
        Validate node configuration.
        Override in subclasses for specific validation.
        """
        return True

    def copy(self) -> 'Node':
        """Create a copy of this node."""
        return self.model_copy() 