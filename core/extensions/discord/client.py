"""
Discord Client Module

This module provides Discord integration for the ChatAgent.
It handles message events and manages the Discord bot lifecycle.

Key Features:
- Discord message handling with typing indicators
- Mention-based triggering
- Message cleaning and formatting
- Long message handling
- Error handling and recovery
"""

import logging
import discord
from typing import Optional
import re
from pydantic import BaseModel

from ai.base.agent import BaseAgent

logger = logging.getLogger(__name__)

class DiscordClient(discord.Client):
    """
    Discord client that integrates with the ChatAgent.
    
    This client:
    - Handles Discord events
    - Routes messages to the ChatAgent
    - Manages bot lifecycle
    - Handles Discord-specific formatting
    """
    
    def __init__(self, agent: BaseAgent, intents: discord.Intents, token: str):
        """
        Initialize the Discord client.
        
        Args:
            agent (BaseAgent): The chat agent instance
            intents (discord.Intents): Discord intents configuration
            token (str): Discord bot token for authentication
        """
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
        
        This handler:
        1. Filters bot messages
        2. Checks for mentions
        3. Cleans message content
        4. Shows typing indicator
        5. Processes response
        6. Handles long messages
        
        Args:
            message (discord.Message): The incoming message
        """
        try:
            # Ignore messages from the bot itself
            if message.author.id == self.user.id:
                return

            # Only respond to mentions
            if not self.user.mentioned_in(message):
                return
                
            # Clean the message content
            cleaned_content = message.content.replace(f"<@{self.user.id}>", "").strip()
            cleaned_content = re.sub(r'<@!?[0-9]+>', '', cleaned_content).strip()
            
            # Skip if content is empty after cleaning
            if not cleaned_content:
                return
            
            # Show typing indicator during processing
            async with message.channel.typing():
                # Process the message through the agent
                response = await self.agent._step(cleaned_content)
                
                # Send the response, handling long messages
                if response:
                    if len(response) > 2000:
                        # Split into chunks of 2000 characters
                        chunks = [response[i:i + 1900] for i in range(0, len(response), 1900)]
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                await message.channel.send(chunk)
                            else:
                                await message.channel.send(f"...{chunk}")
                    else:
                        await message.channel.send(response)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await message.channel.send(
                "I apologize, but I encountered an error while processing your message. "
                "Please try again or rephrase your request."
            )
            
    async def start(self):
        """Start the Discord client with the stored token."""
        await super().start(self.token)