"""
Entry point for running the Eliza Discord bot.
Handles initialization of all components and manages the Discord client lifecycle.
"""

import os
import logging
import asyncio
from dotenv import load_dotenv
import discord
from eliza.agent_runtime import AgentRuntime
from eliza.discord_client import DiscordClient
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """
    Initialize and run the Eliza Discord bot.
    """
    try:
        logger.info("Starting bot initialization...")

        token = os.environ.get("DISCORD_BOT_TOKEN")
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN environment variable not set")

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Set up proper intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.guild_messages = True
        intents.members = True

        # Load agent profile
        with open('eliza_agent/personality_profiles/augie.json', 'r') as f:
            agent_profile = json.load(f)

        # Initialize the agent runtime
        agent_runtime = AgentRuntime()
        await agent_runtime.initialize()
        agent_runtime.set_agent_profile(agent_profile)

        # Start the Discord client
        client = DiscordClient(runtime=agent_runtime, intents=intents)
        
        # Log basic connection info
        logger.info("Starting Discord client...")
        
        # Run the client
        await client.start(token)

    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
