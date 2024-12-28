"""
Simple local chat example using RuntimeConfig and LocalRuntime.

This example demonstrates how to set up a local chat session with a configured agent.
"""

import asyncio
import logging
from dotenv import load_dotenv

from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.prompts.persona import AUG_E
from alchemist.ai.base.tools import ImageGenerationTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format
    handlers=[
        logging.FileHandler('logs/alchemist.log'),  # File handler for full logs
        logging.StreamHandler()  # Console handler for filtered logs
    ]
)

# Filter out httpx logs and other noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

async def main():
    """Run a local chat session with configured runtime."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Configure runtime
        config = RuntimeConfig(
            provider="openpipe",
            persona=AUG_E,
            tools=[ImageGenerationTool],
            platform_config={
                "prompt": "\n\033[94mYou:\033[0m ",  # Blue color for user
                "response_prefix": "\033[92mAssistant:\033[0m "  # Green color for assistant
            }
        )
        
        # Initialize and start local runtime
        runtime = LocalRuntime(config)
        await runtime.start()
        
    except Exception as e:
        logger.error(f"\033[91mError:\033[0m {str(e)}")  # Red color for errors
        raise

if __name__ == "__main__":
    asyncio.run(main())