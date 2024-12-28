"""
Discord Chat Bot Example using RuntimeConfig and DiscordRuntime.

This example demonstrates how to set up a Discord bot using our runtime system.
"""

import os
import logging
import asyncio
from dotenv import load_dotenv

from alchemist.ai.base.runtime import RuntimeConfig
from alchemist.core.extensions.discord.runtime import DiscordRuntime
from alchemist.ai.prompts.persona import AUG_E
from alchemist.ai.base.tools import ImageGenerationTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Initialize and run the Discord bot using runtime system."""
    try:
        # Load environment variables
        load_dotenv()
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
        
        # Configure runtime
        config = RuntimeConfig(
            provider="openpipe",
            persona=AUG_E,
            tools=[ImageGenerationTool],
            platform_config={
                "intents": ["message_content", "guilds", "guild_messages"],
                "activity_type": "listening",
                "activity_name": "mentions"
            }
        )
        
        # Initialize and start Discord runtime
        runtime = DiscordRuntime(
            token=token,
            config=config
        )
        
        logger.info("Starting Discord bot...")
        await runtime.start()
        
    except Exception as e:
        logger.error(f"Error running Discord bot: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 