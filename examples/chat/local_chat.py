"""
Chat Agent Test Script

This script provides a simple way to test the ChatAgent locally.
It sets up logging and environment, then runs the agent in interactive mode.
"""

import asyncio
import logging
from dotenv import load_dotenv

from ai.agents.chat.runtime import ChatRuntime
from ai.prompts.persona import AUG_E
from ai.agents.chat.agent import ChatAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the chat agent in interactive mode."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize chat agent
        agent = ChatAgent(
            provider="openpipe",  # Using OpenAI as default
            persona=AUG_E  # Using our techno-druid persona
        )
        
        # Initialize runtime
        runtime = ChatRuntime(agent=agent)
        
        # Start session and run
        await runtime.start()
        
    except Exception as e:
        logger.error(f"Error in chat session: {str(e)}")
        raise
    finally:
        # Ensure we stop cleanly
        if 'runtime' in locals():
            await runtime.stop()

if __name__ == "__main__":
    asyncio.run(main()) 