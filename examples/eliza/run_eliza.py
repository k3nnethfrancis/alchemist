"""
Eliza Discord Bot Runner

This script sets up and runs the Eliza Discord bot.
It handles environment setup, bot configuration, and error handling.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
import discord

from ai.agents.eliza.agent import ElizaAgent
from core.extensions.discord.eliza_client import ElizaDiscordClient
from core.prompts.persona import AUG_E

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Set up and run the Eliza Discord bot."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Verify required environment variables
        required_vars = [
            "DISCORD_BOT_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Set up Discord intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        
        # Initialize Eliza agent
        agent = ElizaAgent(
            provider="openpipe",  # Using OpenPipe as default
            persona=AUG_E,  # Using our techno-druid persona
            cooldown_seconds=3,  # 3 second cooldown between checks
            scan_interval_seconds=120  # 2 minute scan interval
        )
        
        # Initialize Discord client
        client = ElizaDiscordClient(
            agent=agent,
            intents=intents,
            token=os.getenv("DISCORD_BOT_TOKEN")
        )
        
        # Start the client
        await client.start()
        
    except Exception as e:
        logger.error(f"Error running Eliza bot: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 