"""Local chatbot with Discord reading capabilities.

This example demonstrates using the DiscordReaderTool to read channel history.
Before running this, make sure to:

1. Set up environment:
   ```
   DISCORD_READER_TOKEN=your_reader_bot_token
   ```

2. Start the reader bot service in a separate terminal:
   ```bash
   python -m alchemist.core.extensions.discord.run_reader_bot
   ```

3. Run this chatbot:
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
from pathlib import Path
from dotenv import load_dotenv
import os
import json

from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.prompts.persona import KEN_E
from alchemist.ai.base.tools import CalculatorTool, ImageGenerationTool, DiscordReaderTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Run the local Discord reader chatbot."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Load channel configuration
        config_path = Path("config/channels.json")
        if not config_path.exists():
            raise ValueError(
                "channels.json not found. Please run the reader bot first:\n"
                "python -m alchemist.core.extensions.discord.run_reader_bot"
            )
        
        with open(config_path) as f:
            channel_config = json.load(f)
        
        # Configure the Discord reader tool with channel mapping
        DiscordReaderTool.configure(channel_config["channels"])
        
        # Create runtime configuration
        config = RuntimeConfig(
            provider="openpipe",
            model="gpt-4",
            persona=KEN_E,
            tools=[
                CalculatorTool,
                ImageGenerationTool,
                DiscordReaderTool
            ],
            platform_config={
                "prompt_prefix": "You: ",
                "response_prefix": "Assistant: "
            }
        )
        
        # Initialize and start local runtime
        runtime = LocalRuntime(config)
        
        print("\nDiscord Reader Chatbot")
        print("---------------------")
        print("Try asking things like:")
        print('- "Read the last hour of messages from #ai-news"')
        print('- "Show me what was posted in content-stream in the last 2 days"')
        print('- "Get the last 30 minutes of chat from agent-sandbox"')
        print("\nAvailable channels:")
        
        # Print channels by category using loaded configuration
        for category, channels in channel_config["categories"].items():
            print(f"\n{category}:")
            for channel in channels:
                print(f"  #{channel}")
        
        print("\nType 'exit' or 'quit' to stop")
        print("---------------------")
        
        await runtime.start()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat session terminated by user.") 