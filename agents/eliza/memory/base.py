"""
Base memory classes for the Eliza Discord agent.
Provides the foundation for different types of memory management.

This module defines the base classes for managing different types of agent memory,
including message history, agent knowledge, and other contextual information.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseMemory(ABC):
    """
    Base class for all memory types.
    
    Provides the interface that all memory implementations must follow.
    """
    
    @abstractmethod
    async def get_context(self) -> Dict[str, Any]:
        """
        Retrieve context data for templates.
        
        Returns:
            Dict[str, Any]: Context data formatted for use in templates
        """
        pass

class DummyMemory(BaseMemory):
    """
    Simple in-memory storage implementation for testing.
    
    Attributes:
        memories (Dict[str, List[str]]): Channel-specific memory storage
    """
    
    def __init__(self):
        self.memories: Dict[str, List[str]] = {}
        
    async def get_context(self) -> Dict[str, Any]:
        """
        Get the current memory context.
        
        Returns:
            Dict[str, Any]: Current memories organized by channel
        """
        return {"memories": self.memories}
        
    async def add_memory(self, channel_id: str, memory: str):
        """
        Add a new memory for a specific channel.
        
        Args:
            channel_id (str): The Discord channel ID
            memory (str): The memory to store
        """
        if channel_id not in self.memories:
            self.memories[channel_id] = []
        self.memories[channel_id].append(memory)