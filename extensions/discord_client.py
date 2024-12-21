"""
Discord Client Extension Module

This module implements a Discord bot client that integrates with the ChatAgent system.
It handles Discord-specific interactions and routes messages between Discord and the
agent runtime.

Key Features:
- Discord message handling
- Mention-based triggering
- Message cleaning and formatting
- Asynchronous response handling
- Clean separation from agent logic

Usage:
    runtime = AgentRuntime()
    client = DiscordClient(runtime, intents)
    await client.start(token)

Note:
    This client is designed to be minimal and focused solely on Discord interaction
    handling, delegating all AI/chat logic to the AgentRuntime.
"""

import logging
import discord
from discord.ext import commands
from typing import Optional

logger = logging.getLogger(__name__)

class DiscordClient(commands.Bot):
    """
    A Discord bot client that integrates with the ChatAgent system.
    
    This client handles all Discord-specific functionality while delegating
    message processing to an AgentRuntime instance. It supports mention-based
    interactions and maintains a clean separation between Discord handling and
    agent logic.

    Attributes:
        agent_runtime: The runtime instance that manages the ChatAgent.
                      Must implement a 'get_response' method.
    """

    def __init__(self, agent_runtime, intents: discord.Intents, token: str, command_prefix: str = "!"):
        """
        Initialize the Discord client with necessary configurations.

        Args:
            agent_runtime: The runtime orchestrator that manages the ChatAgent.
            intents (discord.Intents): Discord intents required by the bot.
            token (str): Discord bot token for authentication.
            command_prefix (str): Command prefix for the bot. Defaults to "!".

        Note:
            The client requires message content intents to be enabled
            to read message content.
        """
        super().__init__(command_prefix=command_prefix, intents=intents)
        self.agent_runtime = agent_runtime
        self.token = token  # Store token for start method

    async def on_ready(self):
        """
        Event handler for when the bot has successfully connected to Discord.
        
        This method is called when the bot has:
        1. Successfully connected to Discord
        2. Cached guild data
        3. Prepared for message handling

        Logs the bot's connection status and user information.
        """
        logger.info("Logged in as %s (%d)", self.user.name, self.user.id)

    async def on_message(self, message: discord.Message):
        """
        Event handler for processing new Discord messages.

        This method:
        1. Filters out self-messages
        2. Checks for bot mentions
        3. Cleans message content
        4. Gets agent responses
        5. Sends responses back to Discord

        Args:
            message (discord.Message): The Discord message to process.

        Note:
            Only responds to messages where the bot is mentioned.
            Ignores messages from the bot itself to prevent loops.
        """
        # Ignore messages from ourselves
        if message.author.id == self.user.id:
            return

        # Simple mention check: If the bot is mentioned, respond
        if self.user.mentioned_in(message):
            # Get the raw message content minus the mention
            cleaned_content = message.content.replace(f"@{self.user.name}", "").strip()

            # Obtain a response from our agent runtime
            agent_response = self.agent_runtime.get_response(cleaned_content)

            # Send the agent's message back to Discord
            if agent_response:
                await message.channel.send(agent_response)

    async def start(self):
        """Start the Discord client with the stored token."""
        await super().start(self.token)