"""
Main Application Module

This module serves as the entry point for the application, supporting different
agents and extensions through command-line configuration.

Usage:
    python -m core.main --agent chat --extension discord
    
Note:
    For CLI mode, use: python -m agents.chat.agent directly
"""

import os
import asyncio
import logging
import argparse
from dotenv import load_dotenv
from core.agent_runtime import AgentRuntime, RuntimeConfig
from extensions.config import get_extension_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start an AI agent with specified extension",
        epilog="Note: For CLI mode, use: python -m agents.chat.agent directly"
    )
    parser.add_argument("--agent", choices=["chat"], required=True,
                      help="Type of agent to run")
    parser.add_argument("--extension", choices=["discord"], required=True,
                      help="Extension to use")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="anthropic",
                      help="LLM provider to use")
    parser.add_argument("--model", default="claude-3-5-sonnet-20240620",
                      help="Model to use")
    return parser.parse_args()

async def main():
    """Initialize and run the application based on command line arguments."""
    try:
        args = parse_args()
        
        # Get extension-specific configuration
        extension_config = get_extension_config(args.extension)

        # Initialize runtime with configuration
        config = RuntimeConfig(
            agent_type=args.agent,
            extension=args.extension,
            provider=args.provider,
            model=args.model,
            extension_config=extension_config
        )
        
        runtime = AgentRuntime(config)
        await runtime.start()

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
