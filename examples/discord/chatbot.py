"""Discord chatbot example.

This example demonstrates a Discord bot that:
1. Uses the DiscordRuntime for core functionality
2. Implements a chat agent with tool support
3. Responds to messages in configured channels
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
file = os.path.abspath(__file__)
parent = os.path.dirname(os.path.dirname(os.path.dirname(file)))
sys.path.insert(0, parent)

from alchemist.ai.prompts.persona import KEN_E
from alchemist.core.extensions.discord.runtime import DiscordRuntimeConfig, DiscordRuntime
from alchemist.ai.base.tools import CalculatorTool, ImageGenerationTool
from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime

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

class DiscordChatbot:
    """Discord chatbot with tool support."""
    
    def __init__(self, token: str, channel_ids: list[str]):
        """Initialize the chatbot.
        
        Args:
            token: Discord bot token
            channel_ids: List of channel IDs to monitor
        """
        # Configure Discord runtime
        discord_config = DiscordRuntimeConfig(
            bot_token=token,
            channel_ids=channel_ids,
            platform_config={
                "image_channel": True  # Enable image responses
            }
        )
        self.discord = DiscordRuntime(config=discord_config)
        
        # Configure chat runtime
        chat_config = RuntimeConfig(
            provider="openpipe",
            model="gpt-4o-mini",
            persona=KEN_E,
            tools=[CalculatorTool, ImageGenerationTool]
        )
        self.chat = LocalRuntime(config=chat_config)
        
    async def start(self):
        """Start the chatbot."""
        # Add message handler
        self.discord.add_message_handler(self.handle_message)
        
        # Start both runtimes
        await self.chat.start()
        await self.discord.start()
        
    async def stop(self):
        """Stop the chatbot."""
        await self.discord.stop()
        
    async def handle_message(self, message):
        """Handle incoming Discord messages."""
        logger.info(f"[Discord] Message from {message.author}: {message.content}")
        
        try:
            # Get response from chat runtime
            response = await self.chat.chat(message.content)
            
            # Send response back to Discord
            await message.channel.send(response)
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await message.channel.send("Sorry, I encountered an error processing your message.")

async def main():
    """Run the Discord chatbot."""
    try:
        # Load environment variables
        load_dotenv()
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            logger.error("DISCORD_BOT_TOKEN not set in .env file")
            sys.exit(1)
        
        # Create and start chatbot
        chatbot = DiscordChatbot(
            token=token,
            channel_ids=["1318659602115592204"]  # agent-sandbox channel
        )
        
        logger.info("Starting Discord chatbot...")
        await chatbot.start()
        
        # Keep the bot running
        try:
            await asyncio.Future()  # run forever
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await chatbot.stop()
            
    except Exception as e:
        logger.error(f"Error in Discord bot: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 