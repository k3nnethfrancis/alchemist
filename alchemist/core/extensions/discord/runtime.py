"""Discord Runtime Implementation"""

import discord
from typing import Optional

from alchemist.ai.base.runtime import BaseChatRuntime, RuntimeConfig
from alchemist.core.extensions.discord.client import DiscordClient
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import Graph

class DiscordRuntime(BaseChatRuntime):
    """Runtime for Discord chat interactions."""
    
    def __init__(self, token: str, config: RuntimeConfig, workflow: Optional[Graph] = None):
        super().__init__(config)
        self.token = token
        self.client = None
        self.workflow = workflow
        
    def _create_agent(self):
        """Create agent with Discord-specific configuration."""
        return BaseAgent(
            provider=self.config.provider,
            persona=self.config.persona,
            tools=self.config.tools
        )
    
    async def start(self):
        """Start Discord bot session."""
        self._start_session("discord")
        
        # Setup Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        intents.guilds = True
        
        self.client = DiscordClient(
            agent=self.agent,
            intents=intents,
            token=self.token,
            workflow=self.workflow
        )
        
        await self.client.start()
    
    async def stop(self) -> None:
        """Stop Discord bot session."""
        if self.client:
            await self.client.close() 