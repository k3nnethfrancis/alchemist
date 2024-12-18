"""
Memory Manager Module

Coordinates between different types of memory (message history, agent knowledge)
and provides a unified interface for the agent runtime.
"""

import logging
from typing import Dict, Any, Optional
from discord import Message

from eliza_agent.memory.message_history import MessageHistory
from eliza_agent.memory.agent_knowledge import AgentKnowledge
from core.models.types import RuntimeConfig

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Coordinates different types of memory and provides a unified interface.
    
    Attributes:
        message_history (MessageHistory): Manages chat history
        agent_knowledge (AgentKnowledge): Manages agent's personality and learned info
    """
    
    def __init__(self, runtime_config: RuntimeConfig, agent_profile: Dict[str, Any]):
        self.message_history = MessageHistory()
        self.agent_knowledge = AgentKnowledge(agent_profile)
        
    async def initialize(self):
        """Initialize memory systems."""
        logger.info("Initializing MemoryManager...")
        
    async def get_context(self, channel_id: str) -> Dict[str, Any]:
        """
        Get combined context from all memory sources.
        
        Args:
            channel_id: The Discord channel ID
            
        Returns:
            Combined context from message history and agent knowledge
        """
        message_context = await self.message_history.get_context()
        agent_context = await self.agent_knowledge.get_context()
        
        return {
            "messages": message_context.get("message_histories", {}).get(channel_id, []),
            "agent": agent_context
        }
        
    async def store_interaction(self, message: Message, response: str):
        """Store a new interaction in message history."""
        await self.message_history.add_message(
            str(message.channel.id),
            f"{message.author.name}: {message.content}"
        )
        await self.message_history.add_message(
            str(message.channel.id),
            f"Assistant: {response}"
        )
        
    async def cleanup(self):
        """Cleanup memory systems."""
        logger.info("Cleaning up MemoryManager...")