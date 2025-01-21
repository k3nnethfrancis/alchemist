"""Functional tests for all tools.

This module contains functional tests for all tools in the alchemist package.
One clear test per tool to verify functionality.
"""

import sys
from pathlib import Path
import asyncio
import logging
import os
from dotenv import load_dotenv
import aiohttp
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from alchemist.ai.tools import CalculatorTool, ImageGenerationTool, DiscordReaderTool
from alchemist.core.extensions.discord.runtime import DiscordRuntime, DiscordRuntimeConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# def test_calculator():
#     """Test calculator with a complex expression."""
#     print("\n=== Calculator Tool Test ===")
#     logger.info("Testing Calculator Tool")
#     
#     expr = "(2 + 3) * 4"  # Tests parentheses and multiple operations
#     logger.info(f"Expression: {expr}")
#     tool = CalculatorTool(expression=expr)
#     result = tool.call()
#     logger.info(f"Result: {result}")
#     print(f"{expr} = {result}")

# async def test_image_generation():
#     """Test image generation with a detailed prompt."""
#     print("\n=== Image Generation Tool Test ===")
#     logger.info("Testing Image Generation Tool")
#     
#     prompt = "A serene lake at sunset with mountains in the background"
#     logger.info(f"Prompt: {prompt}")
#     tool = ImageGenerationTool(prompt=prompt)
#     try:
#         result = await tool.call()
#         logger.info(f"Generated image URL: {result}")
#         print(f"Image URL: {result}")
#     except Exception as e:
#         logger.error(f"Error: {str(e)}")
#         print(f"Error: {str(e)}")

async def test_discord_reader():
    """Test the DiscordReaderTool with a natural language query to read messages from a channel."""
    
    logging.info("Starting Discord Reader Tool test...")
    
    # Get channel configuration from bot service
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:5000/channels") as resp:
            config = await resp.json()
            logging.info(f"Got channel configuration: {config}")
    
    # Configure the tool with channel data
    tool = DiscordReaderTool()
    await tool.configure(channels=config["channels"], categories=config["categories"])
    
    # Test reading from ai-news channel
    query = "Read #ai-news"
    result = await tool.call(query)
    
    # Parse the JSON response
    response = json.loads(result)
    messages = response.get("messages", [])
    
    # Count messages with embeds
    embeds_found = sum(1 for msg in messages if msg.get("embeds"))
    logging.info(f"Found {embeds_found} messages with embeds")
    
    # Log a sample message with embeds for verification
    for msg in messages:
        if msg.get("embeds"):
            # Log message details
            logging.info(f"Sample message:")
            logging.info(f"  ID: {msg.get('id')}")
            logging.info(f"  Timestamp: {msg.get('timestamp')}")
            logging.info(f"  Author: {msg.get('author')}")
            logging.info(f"  Content: {msg.get('content')}")
            
            # Log embed details
            for embed in msg.get("embeds", []):
                logging.info("  Embed:")
                logging.info(f"    Title: {embed.get('title')}")
                logging.info(f"    Description: {embed.get('description')}")
                if embed.get("fields"):
                    logging.info("    Fields:")
                    for field in embed.get("fields", []):
                        logging.info(f"      {field.get('name')}: {field.get('value')}")
            break
    
    return result

async def main():
    """Run all functional tests."""
    print("Starting Functional Tests")
    print("=" * 50)
    
    # test_calculator()
    # await test_image_generation()
    await test_discord_reader()
    
    print("\n" + "=" * 50)
    print("Functional tests complete - check logs for details")

if __name__ == "__main__":
    asyncio.run(main()) 