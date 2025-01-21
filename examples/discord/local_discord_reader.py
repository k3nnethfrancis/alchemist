"""Local chatbot with Discord reading capabilities.

This example demonstrates using the DiscordReaderTool to read channel history.
Before running this, make sure to:

1. Start the Discord bot service in another terminal:
   ```bash
   python -m examples.discord.run_bot
   ```

2. Run this chatbot:
   ```bash
   python -m examples.discord.local_discord_reader
   ```

You can then ask the bot to read Discord channels with commands like:
- "Read the last hour of messages from #ai-news"
- "Show me what was posted in content-stream in the last 2 days"
- "Get the last 30 minutes of chat from agent-sandbox"
"""

import asyncio
import logging
import json
from pathlib import Path
import aiohttp
from aiohttp.client_exceptions import ClientConnectorError
from pydantic import BaseModel, Field

from alchemist.ai.tools.calculator import CalculatorTool
from alchemist.ai.tools.image import ImageGenerationTool
from alchemist.ai.tools.discord_tool import DiscordReaderTool
from alchemist.ai.prompts.persona import KEN_E
from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.base.agent import BaseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instance that will be configured
READER_INSTANCE = None

# Subclass DiscordReaderTool to use the configured instance
class ConfiguredDiscordReader(DiscordReaderTool, BaseModel):
    """Discord reader tool that uses a globally configured instance."""
    
    query: str = Field(
        ...,
        description="Natural language query specifying channel and time range"
    )

    async def call(self) -> str:
        """Execute the query using the configured instance."""
        instance = self.get_instance()
        if instance is None:
            raise RuntimeError("Discord reader not configured")
        return await instance.call(self.query)

    @classmethod
    def get_instance(cls):
        return READER_INSTANCE

def print_welcome_message(categories: dict):
    """Print welcome message with available channels.
    
    Args:
        categories: Dict mapping category names to channel names
    """
    print("\nDiscord Reader Chatbot")
    print("---------------------")
    print("Try asking things like:")
    print('- "Read the last hour of messages from #ai-news"')
    print('- "Show me what was posted in content-stream in the last 2 days"')
    print('- "Get the last 30 minutes of chat from agent-sandbox"')
    print("\nAvailable channels:")
    
    # Print channels by category
    for category, channels in categories.items():
        print(f"\n{category}:")
        for channel in channels:
            print(f"  #{channel}")
            
    print("\nType 'exit' or 'quit' to stop")
    print("---------------------")

async def wait_for_bot_service(max_retries: int = 5, retry_delay: float = 2.0):
    """Wait for the bot service to become available.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        
    Raises:
        RuntimeError: If bot service is not available after retries
    """
    logger.info("Checking bot service availability...")
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:5000/channels") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("Successfully connected to bot service")
                        return data
                    
        except ClientConnectorError:
            if attempt < max_retries - 1:
                logger.info(f"Bot service not ready, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            continue
            
    raise RuntimeError(
        "Bot service not running or not responding. "
        "Please start it first with: python -m examples.discord.run_bot"
    )

async def main():
    """Run the local Discord reader."""
    try:
        global READER_INSTANCE
        
        # Check if bot service is running
        logger.info("Checking bot service availability...")
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:5000/channels") as response:
                if response.status != 200:
                    raise RuntimeError("Bot service not available")
                channel_data = await response.json()
                
        logger.info("Successfully connected to bot service")
        
        # Configure Discord reader tool
        READER_INSTANCE = DiscordReaderTool()
        await READER_INSTANCE.configure(
            channels=channel_data["channels"],
            categories=channel_data["categories"]
        )
        
        # Start chat session
        print_welcome_message(channel_data["categories"])
        print("\nStarting chat session. Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                    
                # Process user input with agent
                agent = BaseAgent(
                    tools=[ConfiguredDiscordReader],  # Use the class that knows about our instance
                    persona=KEN_E
                )
                response = await agent._step(user_input)
                print(f"\nA: {response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"\nError: {str(e)}\n")
                
        print("\nChat session ended. Goodbye! ✨")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat session terminated by user.") 