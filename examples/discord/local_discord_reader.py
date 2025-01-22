"""
Simple local chat example using the Discord reader tool.
Before running, make sure to start the Discord bot service:
python -m examples.discord.run_bot
"""

import asyncio
import logging
from pathlib import Path
import aiohttp
from aiohttp.client_exceptions import ClientConnectorError
from dotenv import load_dotenv

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.prompts.persona import KEN_E
from alchemist.ai.tools.discord_tool import DiscordToolKit

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def check_bot_service():
    """Check if the Discord bot service is running."""
    logger.info("Checking bot service availability...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:5000/channels") as response:
                if response.status != 200:
                    raise RuntimeError("Bot service not available")
                data = await response.json()
                logger.info("Successfully connected to bot service")
                return data
    except ClientConnectorError as e:
        raise RuntimeError(
            "Bot service not running. Start it with: python -m examples.discord.run_bot"
        ) from e

async def run_with_agent(channel_data: dict):
    """Example of direct agent initialization and chat."""
    # Initialize Discord toolkit with channel data
    toolkit = DiscordToolKit(
        channels=channel_data["channels"],
        categories=channel_data["categories"]
    )
    
    # Initialize agent with toolkit
    agent = BaseAgent(
        tools=toolkit.create_tools(),
        persona=KEN_E,
        context={
            "channels": channel_data["channels"],
            "categories": channel_data["categories"]
        },
        system_prompt_extension=toolkit.get_system_prompt()
    )
    print("\nChat directly with agent (Ctrl+C to exit)")
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit']:
                print("\nChat session ended. Goodbye! ✨")
                break
            response = await agent._step(query)
            print(f"\nAgent: {response}")
        except KeyboardInterrupt:
            print("\nChat session ended")
            break

async def run_with_runtime(channel_data: dict):
    """Example of using LocalRuntime for a more configured experience."""
    # Initialize Discord toolkit with channel data
    toolkit = DiscordToolKit(
        channels=channel_data["channels"],
        categories=channel_data["categories"]
    )
    
    # Create runtime configuration
    config = RuntimeConfig(
        provider="openpipe",
        model="openpipe:ken0-llama31-8B-instruct",
        persona=KEN_E,
        tools=toolkit.create_tools(),
        context={
            "channels": channel_data["channels"],
            "categories": channel_data["categories"]
        },
        system_prompt_extension=toolkit.get_system_prompt(),
        platform_config={
            "prompt_prefix": "You: ",
            "response_prefix": "Assistant: "
        }
    )
    
    # Initialize and start local runtime
    runtime = LocalRuntime(config)
    print("\nChat using runtime (Ctrl+C to exit)")
    
    await runtime.start()

async def main():
    """Run both chat examples."""
    try:
        load_dotenv()  # Load environment variables
        
        # Check bot service first
        channel_data = await check_bot_service()
        
        # Uncomment one of these to try different approaches:
        await run_with_agent(channel_data)  # Direct agent initialization
        # await run_with_runtime(channel_data)  # Using LocalRuntime
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat session terminated by user.") 