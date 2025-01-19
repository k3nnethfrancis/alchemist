"""Context supplier node implementations."""

import logging
from typing import Dict, Any, Optional, List, Callable
from pydantic import Field
from datetime import datetime

from alchemist.ai.graph.base import Node, NodeState

logger = logging.getLogger(__name__)

class ContextSupplierNode(Node):
    """Base class for nodes that supply context to the graph.
    
    This node type is responsible for gathering and injecting context into the
    graph state. It can be configured with multiple suppliers and will run them
    in parallel by default.
    
    Attributes:
        suppliers: Dictionary of named supplier functions
        parallel: Whether this node can run in parallel (default True)
        target_key: Key in context metadata to store results
    """
    
    suppliers: Dict[str, Callable] = Field(default_factory=dict)
    parallel: bool = True
    target_key: str = "context"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Execute all suppliers and store results in context.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
            
        The node will:
        1. Execute all configured suppliers in parallel
        2. Store results in context metadata under target_key
        3. Return the next node ID based on success/error
        """
        try:
            results = {}
            
            # Execute all suppliers
            for name, supplier in self.suppliers.items():
                try:
                    result = await supplier()
                    results[name] = result
                except Exception as e:
                    logger.error(f"Error in supplier {name}: {str(e)}")
                    results[name] = {"error": str(e)}
            
            # Store results in context
            state.context.metadata[self.target_key] = {
                "data": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store execution details in results
            state.results[self.id] = {
                "suppliers": list(self.suppliers.keys()),
                "target_key": self.target_key,
                "success": True
            }
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            logger.error(f"Error in context supplier node: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "suppliers": list(self.suppliers.keys()),
                "target_key": self.target_key
            }
            return self.next_nodes.get("error")

class TimeContextNode(ContextSupplierNode):
    """Node that supplies time-based context.
    
    This node provides various time-related context values like current time,
    date components, timezone info, etc.
    """
    
    def __init__(self, **data):
        """Initialize with time suppliers."""
        super().__init__(**data)
        
        # Add default time suppliers
        self.suppliers = {
            "current_time": lambda: datetime.now().isoformat(),
            "timestamp": lambda: datetime.now().timestamp(),
            "date": lambda: datetime.now().date().isoformat(),
            "year": lambda: datetime.now().year,
            "month": lambda: datetime.now().month,
            "day": lambda: datetime.now().day,
            "hour": lambda: datetime.now().hour,
            "minute": lambda: datetime.now().minute,
            "weekday": lambda: datetime.now().strftime("%A")
        }
        self.target_key = "time_context"

class MemoryContextNode(ContextSupplierNode):
    """Node that supplies memory-based context.
    
    This node provides access to the graph's memory system and can be configured
    to retrieve specific memory entries or search for relevant context.
    
    Attributes:
        memory_keys: List of keys to retrieve from memory
        search_query: Optional query to search memory
    """
    
    memory_keys: List[str] = Field(default_factory=list)
    search_query: Optional[str] = None
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Retrieve memory context.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
        """
        try:
            results = {}
            
            # Get specific memory entries
            for key in self.memory_keys:
                results[key] = state.context.memory.get(key)
            
            # Perform memory search if query provided
            if self.search_query:
                # TODO: Implement memory search
                results["search"] = {
                    "query": self.search_query,
                    "results": []  # Placeholder for search results
                }
            
            # Store results in context
            state.context.metadata["memory_context"] = {
                "data": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store execution details
            state.results[self.id] = {
                "memory_keys": self.memory_keys,
                "search_query": self.search_query,
                "success": True
            }
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            logger.error(f"Error in memory context node: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "memory_keys": self.memory_keys,
                "search_query": self.search_query
            }
            return self.next_nodes.get("error") 