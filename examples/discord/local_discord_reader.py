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
        
        # Configure channel mapping for agency42 server
        channel_map = {
            "general": "1288645478975541280",
            "resources": "1288645711155564585",
            "content-stream": "1310110972500774943",
            "agency42": "1310111072690110534",
            "agent-sandbox": "1318659602115592204",
            "creative-riffs": "1319706555070939177",
            "clients": "1321948182946512956",
            "projects": "1322278531253801011",
            "memecoins": "1322953493161574482",
            "whiteboard": "1323027435725520926",
            "ai-memories": "1323417443703455754",
            "infra": "1325985507590799430",
            "ai-news": "1326422578340036689",
            "action-list": "1327092216807690352"
        }
        
        # Configure the Discord reader tool with channel mapping
        DiscordReaderTool.configure(channel_map)
        
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
        
        # Print channels by category
        categories = {
            "augmented builders": [
                "general", "resources", "content-stream", "agent-sandbox",
                "memecoins", "whiteboard", "ai-memories", "ai-news"
            ],
            "Core": [
                "agency42", "creative-riffs", "clients", "projects",
                "infra", "action-list"
            ]
        }
        
        for category, channels in categories.items():
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