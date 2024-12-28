"""Eliza Discord Bot

This script demonstrates running the Eliza agent as a Discord bot.
The bot uses a graph-based workflow to:
1. Monitor chat history
2. Analyze conversation context
3. Generate contextually appropriate responses
"""

import os
import discord
import logging
import asyncio
from dotenv import load_dotenv

from alchemist.ai.agents.eliza.agent import ElizaAgent
from alchemist.ai.prompts.persona import ELIZA
from alchemist.core.extensions.discord.client import DiscordClient
from alchemist.core.extensions.config import get_discord_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Initialize and run the Eliza Discord bot."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Discord configuration
        config = get_discord_config()
        
        # Initialize Eliza agent
        agent = ElizaAgent(
            provider="openai",  # Using OpenAI for development
            persona=ELIZA  # Using Eliza persona
        )
        
        # Initialize Discord client
        client = DiscordClient(
            agent=agent,
            intents=config["intents"],
            token=config["token"]
        )
        
        # Start the bot
        logger.info("Starting Eliza Discord bot...")
        await client.start()
        
    except Exception as e:
        logger.error(f"Error running Eliza Discord bot: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 