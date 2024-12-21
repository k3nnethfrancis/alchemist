"""
Discord Client Module

This module sets up the Discord client, handles events, and interacts with the AgentRuntime for message processing.

Key Components:
    - Event Handling: Manages Discord events (messages, reactions, etc.)
    - Runtime Integration: Coordinates with AgentRuntime for message processing
    - Scheduled Tasks: Runs periodic analysis tasks
"""

import logging
import discord
from discord.ext import commands
from typing import Optional

from eliza.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)

class DiscordClient(commands.Bot):
    """
    Discord client that handles bot interactions and message processing.
    
    Attributes:
        runtime (AgentRuntime): The agent runtime instance for message processing
    """
    
    def __init__(self, runtime: AgentRuntime, intents: discord.Intents):
        # Initialize the commands.Bot with intents
        super().__init__(
            command_prefix="!",  # Default prefix, though we won't use commands
            intents=intents
        )
        self.runtime = runtime

    async def setup_hook(self):
        """Called when the client is done preparing data"""
        logger.info("Bot is ready and setting up...")
        # Start the periodic analysis task
        self.loop.create_task(self.runtime.periodic_analysis(self))

    async def on_ready(self):
        """Called when the client is done preparing data after login"""
        logger.info(f"Logged in as {self.user.name} ({self.user.id})")
        self.runtime.bot_user_id = self.user.id

    async def on_message(self, message: discord.Message):
        """
        Handles incoming messages and generates responses through the runtime.
        
        Args:
            message (discord.Message): The incoming Discord message
        """
        try:
            # Ignore messages from the bot itself
            if message.author.id == self.user.id:
                return

            # Add the message to the message history
            await self.runtime.message_history.add_message(
                str(message.channel.id),
                f"{message.author.name}: {message.content}"
            )

            # Get context and determine if we should respond
            context = await self.runtime.message_history._get_channel_messages(str(message.channel.id))
            should_respond = await self.runtime.should_respond(
                name=self.runtime.agent_profile["name"],
                bio=self.runtime.agent_profile["bio"],
                context="\n".join(context),
                message=message.content
            )

            if "RESPOND" in should_respond.content.upper():
                response = await self.runtime.generate_response(
                    name=self.runtime.agent_profile["name"],
                    bio=self.runtime.agent_profile["bio"],
                    style_guidelines="\n".join(self.runtime.agent_profile["style"]["chat"]),
                    context="\n".join(context),
                    message=message.content
                )
                
                await message.channel.send(response.content)
                await self.runtime.message_history.add_message(
                    str(message.channel.id),
                    f"{self.runtime.agent_profile['name']}: {response.content}"
                )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")