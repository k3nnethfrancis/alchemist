"""
Discord Client Module

This module provides Discord integration for the ChatAgent.
"""

import logging
import discord
from typing import Optional
import re
from datetime import datetime
from pydantic import BaseModel

from alchemist.ai.base.agent import BaseAgent

logger = logging.getLogger(__name__)

class DiscordClient(discord.Client):
    """Discord client implementation for chat agents."""
    
    def __init__(self, agent: BaseAgent, intents: discord.Intents, token: str):
        """Initialize the Discord client."""
        super().__init__(intents=intents)
        self.agent = agent
        self.token = token

    async def setup_hook(self):
        """Called when the client is done preparing data."""
        logger.info("Bot is ready and setting up...")

    async def on_ready(self):
        """Called when the client is done preparing data after login."""
        logger.info(f"Logged in as {self.user.name} ({self.user.id})")

    async def on_message(self, message: discord.Message):
        """
        Handle incoming Discord messages.
        
        Args:
            message (discord.Message): The incoming message
        """
        try:
            # Ignore messages from bots (including self)
            if message.author.bot:
                logger.debug("Skipping message from bot")
                return

            # Check if message is a mention
            is_mention = self.user.mentioned_in(message)
            
            # Clean the message content if it's a mention
            content = message.content
            if is_mention:
                content = content.replace(f"<@{self.user.id}>", "").strip()
                content = re.sub(r'<@!?[0-9]+>', '', content).strip()
            
            # Skip if content is empty after cleaning
            if not content:
                return
            
            # Convert to agent message format
            agent_message = {
                "content": content,
                "bot_id": str(self.user.id),
                "author": {
                    "id": str(message.author.id),
                    "name": message.author.name,
                    "bot": message.author.bot
                },
                "channel": {
                    "id": str(message.channel.id),
                    "name": message.channel.name
                },
                "mentions": [
                    {
                        "id": str(mention.id),
                        "name": mention.name,
                        "bot": mention.bot
                    }
                    for mention in message.mentions
                ],
                "timestamp": datetime.timestamp(message.created_at)
            }
            
            # Process through agent
            logger.info(f"Processing message: {content}")
            if hasattr(self.agent, 'process_discord_message'):
                # Use specialized Discord processing if available
                response = await self.agent.process_discord_message(agent_message)
            else:
                # Only process standard agents when mentioned
                if not is_mention:
                    logger.debug("Skipping message - bot not mentioned")
                    return
                # Fall back to standard processing
                response = await self.agent._step(content)
            
            # Send response if one was generated
            if response:
                logger.info(f"Sending response: {response}")
                await message.channel.send(response)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing message: {error_msg}", exc_info=True)
            await message.channel.send(
                f"I apologize, but I encountered an error while processing your message: {error_msg}. "
                "Please try again or rephrase your request."
            )
            
    async def start(self):
        """Start the Discord client with the stored token."""
        await super().start(self.token)