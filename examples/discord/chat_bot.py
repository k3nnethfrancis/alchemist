"""
Example Discord Bot Runner

This script demonstrates how to set up and run a Discord bot using the ChatAgent.
It provides a complete example of bot configuration and initialization.
"""

import os
import discord
import logging
import asyncio
from dotenv import load_dotenv
from ai.agents.chat.agent import ChatAgent
from ai.prompts.persona import AUG_E
from core.extensions.discord.client import DiscordClient
from core.extensions.config import get_discord_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Initialize and run the Discord bot."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Discord configuration
        config = get_discord_config()
        
        # Initialize chat agent
        agent = ChatAgent(
            provider="openpipe",  # Using OpenPipe as default
            persona=AUG_E  # Using our techno-druid persona
        )
        
        # Initialize Discord client
        client = DiscordClient(
            agent=agent,
            intents=config["intents"],
            token=config["token"]
        )
        
        # Start the bot
        logger.info("Starting Discord bot...")
        await client.start()
        
    except Exception as e:
        logger.error(f"Error running Discord bot: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 