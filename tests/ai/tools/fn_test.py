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
from datetime import datetime, timedelta, timezone

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from alchemist.ai.tools import CalculatorTool, ImageGenerationTool, DiscordTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_calculator():
    """Test calculator with a complex expression."""
    print("\n=== Calculator Tool Test ===")
    logger.info("Testing Calculator Tool")
    
    expr = "(2 + 3) * 4"  # Tests parentheses and multiple operations
    logger.info(f"Expression: {expr}")
    tool = CalculatorTool(expression=expr)
    result = tool.call()
    logger.info(f"Result: {result}")
    print(f"{expr} = {result}")

async def test_image_generation():
    """Test image generation with a detailed prompt."""
    print("\n=== Image Generation Tool Test ===")
    logger.info("Testing Image Generation Tool")
    
    prompt = "A serene lake at sunset with mountains in the background"
    logger.info(f"Prompt: {prompt}")
    tool = ImageGenerationTool(prompt=prompt)
    try:
        result = await tool.call()
        logger.info(f"Generated image URL: {result}")
        print(f"Image URL: {result}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")

async def test_discord():
    """Test Discord toolkit with real channel data."""
    print("\n=== Discord Toolkit Test ===")
    logger.info("Testing Discord Toolkit")
    
    # Load channel data from config
    config_path = project_root / "config" / "channels.json"
    with open(config_path) as f:
        data = json.load(f)
        channels = data["channels"]
        categories = data["categories"]
    
    logger.info(f"Found {len(channels)} channels in {len(categories)} categories")
    print("\nAvailable channels:")
    for name, id in channels.items():
        print(f"  #{name}: {id}")
    
    # Create toolkit instance
    toolkit = DiscordTools(
        channels=channels,
        categories=categories
    )
    
    # Test reading from ai-news channel
    channel_name = "ai-news"
    print(f"\nTrying to read #{channel_name}")
    
    try:
        messages = await toolkit.read_channel(
            channel_name=channel_name,
            limit=5
        )
        print(f"Retrieved {len(messages)} messages")
        if messages:
            msg = messages[0]
            print("Latest message:")
            print(f"  Author: {msg['author']}")
            print(f"  Content: {msg['content'][:100]}...")
            if msg["embeds"]:
                print(f"  Embeds: {len(msg['embeds'])}")
                embed = msg["embeds"][0]
                print(f"    Title: {embed.get('title')}")
                print(f"    Description: {embed.get('description', '')[:100]}...")
    except Exception as e:
        print(f"Error reading {channel_name}: {str(e)}")
    
    # Test reading from agent-sandbox
    channel_name = "agent-sandbox"
    print(f"\nTrying to read #{channel_name}")
    
    try:
        messages = await toolkit.read_channel(
            channel_name=channel_name,
            limit=5
        )
        print(f"Retrieved {len(messages)} messages")
        if messages:
            msg = messages[0]
            print("Latest message:")
            print(f"  Author: {msg['author']}")
            print(f"  Content: {msg['content'][:100]}...")
    except Exception as e:
        print(f"Error reading {channel_name}: {str(e)}")

async def main():
    """Run all functional tests."""
    print("Starting Functional Tests")
    print("=" * 50)
    
    test_calculator()
    await test_image_generation()
    await test_discord()
    
    print("\n" + "=" * 50)
    print("Functional tests complete - check logs for details")

if __name__ == "__main__":
    asyncio.run(main()) 