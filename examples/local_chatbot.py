"""
Simple local chat example demonstrating two approaches:
1. Direct BaseAgent initialization
2. LocalRuntime for a more configured experience

Includes both calculator and image generation capabilities.
"""

import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
# from alchemist.ai.prompts.persona import KEN_E
from alchemist.ai.tools.calculator import CalculatorTool
from alchemist.ai.tools.image import ImageGenerationTool

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def run_with_agent():
    """Example of direct agent initialization and chat."""
    # Initialize agent with both tools
    agent = BaseAgent(
        tools=[CalculatorTool, ImageGenerationTool],
        # persona=KEN_E
    )
    
    print("\nChat directly with agent (Ctrl+C to exit)")
    print("Try asking for calculations or image generation!")
    print("----------------------------------------")
    
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

async def run_with_runtime():
    """Example of using LocalRuntime for a more configured experience."""
    # Create runtime configuration
    config = RuntimeConfig(
        provider="openpipe",
        model="openpipe:ken0-llama31-8B-instruct",
        # persona=KEN_E,
        tools=[CalculatorTool, ImageGenerationTool],
        platform_config={
            "prompt_prefix": "You: ",
            "response_prefix": "Assistant: "
        }
    )
    
    # Initialize and start local runtime
    runtime = LocalRuntime(config)
    print("\nChat using runtime (Ctrl+C to exit)")
    print("Try asking for calculations or image generation!")
    print("-----------------------------------")
    
    await runtime.start()

async def main():
    """Run both chat examples."""
    try:
        load_dotenv()  # Load environment variables
        
        # Uncomment one of these to try different approaches:
        # await run_with_agent()  # Direct agent initialization
        await run_with_runtime()  # Using LocalRuntime
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat session terminated by user.")