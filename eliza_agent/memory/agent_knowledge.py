"""
Agent knowledge memory management for the Eliza Discord agent.
Manages the agent's personality, behavioral patterns, and learned information.

This module handles the storage and retrieval of agent-specific knowledge,
including personality traits, behavioral guidelines, and accumulated knowledge.
"""

from typing import Dict, Any
from .base import BaseMemory

class AgentKnowledge(BaseMemory):
    """
    Manages agent-specific knowledge and personality information.
    
    Stores and provides access to the agent's personality configuration,
    behavioral patterns, and accumulated knowledge.
    
    Attributes:
        agent_profile (Dict[str, Any]): The agent's personality profile
        learned_info (Dict[str, Any]): Information learned during interactions
    """
    
    def __init__(self, agent_profile: Dict[str, Any]):
        """
        Initialize the agent knowledge manager.
        
        Args:
            agent_profile (Dict[str, Any]): The agent's personality configuration
        """
        self.agent_profile = agent_profile
        self.learned_info: Dict[str, Any] = {}
        
    async def get_context(self) -> Dict[str, Any]:
        """
        Get the current agent knowledge context.
        
        Returns:
            Dict[str, Any]: Combined personality and learned information
        """
        return {
            "bio": self.agent_profile.get('bio', ''),
            "lore": self.agent_profile.get('lore', []),
            "style": self.agent_profile.get('style', {}),
            "name": self.agent_profile.get('name', 'Agent'),
            "learned": self.learned_info
        }
        
    async def add_learned_info(self, key: str, value: Any):
        """
        Add new learned information to the agent's knowledge.
        
        Args:
            key (str): The category or type of information
            value (Any): The information to store
        """
        self.learned_info[key] = value