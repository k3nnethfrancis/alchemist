"""Local chatbot with Discord reading capabilities."""

import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
import discord

from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.prompts.persona import KEN_E
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.base.tools import DiscordReaderTool
from alchemist.ai.base.agent import BaseAgent

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_dir / "alchemist.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Run a local chat session with Discord reading capabilities."""
    try:
        load_dotenv()
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN not set")
        
        # Setup Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        client = discord.Client(intents=intents)
        
        # Start client in background
        asyncio.create_task(client.start(token))
        while not client.is_ready():
            await asyncio.sleep(0.1)
        
        # Create agent with Discord reader tool
        agent = BaseAgent(
            provider="openpipe",
            model="gpt-4",
            persona=PersonaConfig(**KEN_E),
            tools=[DiscordReaderTool(client=client)]
        )
        
        # Use built-in run method
        agent.run()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat session terminated by user.")