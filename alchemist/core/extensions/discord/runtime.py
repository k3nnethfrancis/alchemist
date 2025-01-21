"""Discord Runtime.

This module provides the runtime implementation for Discord integration,
building on top of the base Discord client to provide higher-level functionality.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
import asyncio
import discord
from pydantic import BaseModel, Field

from alchemist.core.extensions.discord.client import DiscordClient

logger = logging.getLogger(__name__)

class DiscordRuntimeConfig(BaseModel):
    """Configuration for Discord runtime.
    
    Attributes:
        bot_token: Discord bot token for authentication
        channel_ids: List of channel IDs to monitor (["*"] for all)
        platform_config: Additional Discord-specific configuration
    """
    
    bot_token: str = Field(..., description="Discord bot token for authentication")
    channel_ids: List[str] = Field(
        default=["*"],
        description="List of Discord channel IDs to monitor ('*' for all)"
    )
    platform_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional Discord-specific configuration"
    )

class DiscordRuntime:
    """Runtime for Discord integration.
    
    This runtime provides high-level functionality built on top of the Discord client:
    - Channel monitoring and message handling
    - Message history access
    - Channel information retrieval
    """
    
    def __init__(self, config: DiscordRuntimeConfig):
        """Initialize the Discord runtime.
        
        Args:
            config: Runtime configuration
        """
        self.config = config
        self.client = DiscordClient(token=config.bot_token)
        self._task: Optional[asyncio.Task] = None
        
    def add_message_handler(self, handler: Callable[[discord.Message], Awaitable[None]]):
        """Add a message handler function.
        
        Args:
            handler: Async function that takes a discord.Message parameter
        """
        self.client.add_message_handler(handler)
    
    async def start(self):
        """Start the Discord runtime.
        
        This method starts the Discord client in a background task and waits
        for it to be ready before returning.
        """
        logger.info("Starting Discord runtime...")
        
        # Create background task for client
        self._task = asyncio.create_task(self._run_client())
        
        # Wait for client to be ready
        await self.client.ready.wait()
        logger.info("Discord runtime started successfully")
        
    async def stop(self):
        """Stop the Discord runtime.
        
        This method gracefully shuts down the Discord client and cleans up resources.
        """
        logger.info("Stopping Discord runtime...")
        
        if self._task:
            try:
                await self.client.close()
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                logger.error(f"Error stopping client: {str(e)}")
                
        logger.info("Discord runtime stopped")
    
    async def _run_client(self):
        """Run the Discord client in the background."""
        try:
            await self.client.start()
        except Exception as e:
            logger.error(f"Error in client task: {str(e)}")
            raise
    
    async def get_channels(self) -> Dict[str, Dict[str, Any]]:
        """Get all available channels grouped by category.
        
        Returns:
            Dict with:
                channels: Dict[channel_name, channel_id]
                categories: Dict[category_name, List[channel_name]]
        """
        return await self.client.get_channels()
    
    async def get_message_history(
        self,
        channel_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get message history from a channel.
        
        Args:
            channel_id: Discord channel ID
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with id, author, content, and timestamp
            
        Raises:
            ValueError: If channel not found
        """
        return await self.client.get_message_history(channel_id, limit) 