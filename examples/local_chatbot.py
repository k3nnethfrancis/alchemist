"""
Simple local chat example using RuntimeConfig and LocalRuntime.

This example demonstrates how to set up a local chat session with a configured agent.
"""

import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.prompts.persona import KEN_E, AUG_E
from alchemist.ai.prompts.base import PersonaConfig

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_dir / "alchemist.log"),
        logging.StreamHandler()
    ]
)

# Filter noisy loggers
for logger_name in ["httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def main():
    """Run a local chat session with configured runtime."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Create runtime configuration
        config = RuntimeConfig(
            provider="openpipe",
            model="openpipe:ken0-llama31-8B-instruct",
            persona=PersonaConfig(**KEN_E),  # Use Augie as our default persona
            tools=[],  # Start simple without tools
            platform_config={}  # Use default console formatting
        )
        
        # Initialize and start local runtime
        runtime = LocalRuntime(config)
        await runtime.start()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat session terminated by user.")