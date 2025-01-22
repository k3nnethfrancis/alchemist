"""Local Discord reader example.

This example demonstrates using the Discord toolkit to read channel messages
through an interactive chat interface.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.prompts.persona import KEN_E
from alchemist.ai.tools import DiscordTools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_with_agent():
    """Run chat example with direct agent initialization."""
    print("\nStarting Discord Reader Chat...")
    
    # Load channel data from config
    config_path = Path(__file__).parent.parent.parent / "config" / "channels.json"
    with open(config_path) as f:
        config = json.load(f)
        channels = config["channels"]
        categories = config["categories"]
        print(f"\nFound {len(channels)} channels in {len(categories)} categories\n")
        print("Available channels:")
        for channel in channels:
            print(f"  #{channel}")
    
    # Initialize Discord toolkit and get its tools
    discord_tools = DiscordTools(channels=channels, categories=categories)
    
    # Initialize agent with Discord toolkit functions
    agent = BaseAgent(
        tools=discord_tools.create_tools(),  # Create the actual tools from the toolkit
        persona=KEN_E,
        provider="openpipe",
        model="gpt-4o-mini",
    )
    
    print("\nChat with me! I can read messages from Discord channels.")
    print('Type "exit", "quit", or send an empty message to end the chat.')
    print("\nExample commands:")
    print('- "What are the latest messages in #ai-news?"')
    print('- "Show me the last 5 messages from #general"')
    print('- "Summarize recent updates from #resources"\n')
    
    # Start chat loop
    while True:
        try:
            query = input("\nUser: ")
            if not query or query.lower() in ["exit", "quit"]:
                break
            response = await agent._step(query)
            print(f"\nAgent: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

def run_with_runtime():
    """Run chat example with LocalRuntime for more configuration."""
    print("\nStarting Discord Reader Chat (with runtime)...")
    
    # Load channel data from config
    config_path = Path(__file__).parent.parent.parent / "config" / "channels.json"
    with open(config_path) as f:
        data = json.load(f)
        channels = data["channels"]
        categories = data["categories"]
        print(f"\nFound {len(channels)} channels in {len(categories)} categories")
        print("\nAvailable channels:")
        for name in channels:
            print(f"  #{name}")
    
    # Initialize Discord toolkit and get its tools
    toolkit = DiscordTools(
        channels=channels,
        categories=categories
    )
    
    # Configure runtime
    config = RuntimeConfig(
        provider="openpipe",
        model="gpt-4o-mini",
        persona=KEN_E,
        tools=toolkit.create_tools()  # Create the actual tools from the toolkit
    )
    
    # Initialize runtime
    runtime = LocalRuntime(config=config)
    runtime.start()
    
    print("\nChat with me! I can read messages from Discord channels.")
    print('Type "exit", "quit", or send an empty message to end the chat.')
    print("\nExample commands:")
    print('- "What are the latest messages in #ai-news?"')
    print('- "Show me the last 5 messages from #general"')
    print('- "Summarize recent updates from #resources"\n')
    
    while True:
        try:
            # Get user input
            query = input("\nYou: ").strip()
            if not query or query.lower() in ["exit", "quit"]:
                break
                
            # Get runtime response
            response = runtime.chat(query)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            break
    
    print("\nChat ended. Goodbye!")
    runtime.stop()

if __name__ == "__main__":
    # Run both chat examples
    asyncio.run(run_with_agent())
    print("\n" + "="*50 + "\n")
    run_with_runtime() 