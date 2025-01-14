"""Discord chatbot example using the Alchemist framework.

This example demonstrates how to create a Discord bot that can:
1. Respond to mentions
2. Use tools (Calculator and ImageGeneration)
3. Process messages with rich metadata
"""

import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

from alchemist.ai.base.runtime import RuntimeConfig, DiscordRuntime
from alchemist.ai.prompts.persona import KEN_E
from alchemist.ai.base.tools import CalculatorTool, ImageGenerationTool

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "discord_bot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Run the Discord bot."""
    try:
        # Load environment variables
        load_dotenv()
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            logger.error("DISCORD_BOT_TOKEN not set in .env file")
            sys.exit(1)
        
        # Configure runtime
        config = RuntimeConfig(
            provider="openpipe",
            model="gpt-4",
            persona=KEN_E,
            tools=[CalculatorTool, ImageGenerationTool],
            platform_config={
                "image_channel": True  # Enable image responses
            }
        )
        
        # Create and start runtime
        runtime = DiscordRuntime(config=config, token=token)
        
        logger.info("Starting Discord bot...")
        await runtime.start()
        
        # Keep the bot running
        try:
            await asyncio.Future()  # run forever
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await runtime.stop()
            
    except Exception as e:
        logger.error(f"Error in Discord bot: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 