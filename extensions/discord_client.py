"""
Discord Client Extension Module

This module implements a Discord bot client that integrates with the ChatAgent system.
It handles Discord-specific interactions and routes messages between Discord and the
agent runtime, with special handling for tool-based conversations.

Key Features:
- Discord message handling with typing indicators
- Mention-based triggering
- Message cleaning and formatting
- Tool response aggregation
- Asynchronous response handling
- Error handling and recovery

Usage:
    runtime = AgentRuntime()
    client = DiscordClient(runtime, intents)
    await client.start(token)

Note:
    This client is designed to handle multi-step conversations where the agent
    may need to use tools or perform multiple actions before providing a complete
    response.
"""

import logging
import discord
from discord.ext import commands
from typing import Optional, List

logger = logging.getLogger(__name__)

class DiscordClient(commands.Bot):
    """
    A Discord bot client that integrates with the ChatAgent system.
    
    This client handles all Discord-specific functionality while delegating
    message processing to an AgentRuntime instance. It supports mention-based
    interactions and maintains a clean separation between Discord handling and
    agent logic.

    The client is specifically designed to handle tool-based conversations where
    multiple responses may be generated during a single interaction.

    Attributes:
        agent_runtime: The runtime instance that manages the ChatAgent.
                      Must implement get_response and process_tool_result methods.
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
        self.token = token

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

        Handles the complete conversation flow including:
        1. Message filtering and cleaning
        2. Initial response generation
        3. Tool execution and follow-up responses
        4. Error handling and recovery

        Args:
            message (discord.Message): The Discord message to process.

        Note:
            Only responds to messages where the bot is mentioned.
            Ignores messages from the bot itself to prevent loops.
            Shows typing indicator during processing.
        """
        # Ignore messages from ourselves
        if message.author.id == self.user.id:
            return

        # Only respond to mentions
        if self.user.mentioned_in(message):
            # Clean the message content
            cleaned_content = message.content.replace(f"@{self.user.name}", "").strip()
            
            # Show typing indicator during processing
            async with message.channel.typing():
                try:
                    # Get initial response - now properly awaited
                    response = await self.agent_runtime.get_response(cleaned_content)
                    if response:
                        await message.channel.send(response)
                        
                except Exception as e:
                    logger.error("Error processing message: %s", str(e))
                    await message.channel.send(
                        "I apologize, but I encountered an error while processing your message. "
                        "Please try again or rephrase your request."
                    )

    async def start(self):
        """
        Start the Discord client with the stored token.
        
        This method initializes the connection to Discord and begins
        processing events.
        """
        await super().start(self.token)