"""
Eliza Discord Client Module

This module provides Discord integration for the ElizaAgent.
It handles message events, maintains chat history, and manages the bot lifecycle.

Key Features:
- Message history tracking
- Proactive engagement
- Channel context management
- Periodic scanning
"""

import logging
import discord
from discord.ext import commands, tasks
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel

from ai.agents.eliza.agent import ElizaAgent

logger = logging.getLogger(__name__)

class ElizaDiscordClient(commands.Bot):
    """
    Discord client that integrates with the ElizaAgent.
    
    This client:
    - Tracks message history
    - Manages proactive engagement
    - Handles Discord events
    - Maintains channel context
    """
    
    def __init__(self, agent: ElizaAgent, intents: discord.Intents, token: str):
        """
        Initialize the Discord client.
        
        Args:
            agent (ElizaAgent): The Eliza agent instance
            intents (discord.Intents): Discord intents configuration
            token (str): Discord bot token for authentication
        """
        super().__init__(
            command_prefix="!",  # Default prefix, though we mainly use mentions
            intents=intents
        )
        self.agent = agent
        self.token = token
        
        # Message history per channel
        self.channel_history: Dict[int, List[dict]] = {}
        self.channel_contexts: Dict[int, str] = {}
        
        # For testing purposes
        self._test_user: Optional[discord.User] = None
    
    @property
    def user(self) -> Optional[discord.User]:
        """Get the bot's user object."""
        if super().user is None and self._test_user is not None:
            return self._test_user
        return super().user
    
    @user.setter
    def user(self, value: discord.User):
        """Set the bot's user object (for testing only)."""
        self._test_user = value
    
    def get_channel_context(self, channel: discord.TextChannel) -> str:
        """Get or create context description for a channel."""
        if channel.id not in self.channel_contexts:
            # Create basic context from channel info
            context = f"Channel: #{channel.name}\n"
            if channel.topic:
                context += f"Topic: {channel.topic}\n"
            if channel.category:
                context += f"Category: {channel.category.name}\n"
            self.channel_contexts[channel.id] = context
        
        return self.channel_contexts[channel.id]
    
    def format_message(self, message: discord.Message) -> dict:
        """Format a Discord message for the agent."""
        return {
            "id": message.id,
            "author": message.author.name,
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
            "mentions": [str(user) for user in message.mentions],
            "referenced_message": (
                str(message.reference.resolved.content) 
                if message.reference and message.reference.resolved 
                else None
            )
        }
    
    async def setup_hook(self):
        """Called when the client is done preparing data."""
        logger.info("Bot is ready and setting up...")
    
    async def on_ready(self):
        """Called when the client is done preparing data after login."""
        logger.info(f"Logged in as {self.user.name} ({self.user.id})")
    
    @tasks.loop(minutes=2)
    async def scan_channels(self):
        """Periodically scan channels for engagement opportunities."""
        try:
            for channel_id, messages in self.channel_history.items():
                if not messages:
                    continue
                    
                channel = self.get_channel(channel_id)
                if not channel:
                    continue
                
                context = self.get_channel_context(channel)
                response = await self.agent.process_new_messages(messages, context)
                
                if response:
                    await channel.send(response)
                    
        except Exception as e:
            logger.error(f"Error scanning channels: {str(e)}")
    
    async def on_message(self, message: discord.Message):
        """
        Handle incoming Discord messages.
        
        This handler:
        1. Updates message history
        2. Processes mentions
        3. Triggers agent response checks
        
        Args:
            message (discord.Message): The incoming message
        """
        try:
            # Ignore messages from the bot itself
            if message.author.id == self.user.id:
                return
            
            # Update channel history
            channel_id = message.channel.id
            if channel_id not in self.channel_history:
                self.channel_history[channel_id] = []
            
            # Add formatted message to history
            self.channel_history[channel_id].append(self.format_message(message))
            
            # Keep only last 50 messages
            self.channel_history[channel_id] = self.channel_history[channel_id][-50:]
            
            # If mentioned, respond immediately
            if self.user.mentioned_in(message):
                context = self.get_channel_context(message.channel)
                response = await self.agent.process_new_messages(
                    self.channel_history[channel_id],
                    context
                )
                if response:
                    await message.channel.send(response)
            
            # Otherwise, check if we should respond based on recent messages
            else:
                context = self.get_channel_context(message.channel)
                response = await self.agent.process_new_messages(
                    self.channel_history[channel_id][-5:],  # Look at last 5 messages
                    context
                )
                if response:
                    await message.channel.send(response)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            if message.channel.permissions_for(message.guild.me).send_messages:
                await message.channel.send(
                    "I apologize, but I encountered an error while processing the messages. "
                    "Please try again or rephrase your request."
                )
    
    async def start(self):
        """Start the Discord client with the stored token."""
        await super().start(self.token) 