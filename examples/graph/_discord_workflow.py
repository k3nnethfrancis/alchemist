"""
Example of a proactive Discord bot using the updated graph system.

This script sets up a Discord bot that runs a small Alchemist graph workflow.
It uses AgentNode instead of the deprecated LLMNode and includes optional
analysis or decision steps in the pipeline. The bot can proactively respond
to messages, even when not explicitly mentioned.

Usage:
    1. Install dependencies from requirements.txt or poetry.
    2. Set your Discord token in an environment variable or .env file.
    3. Run: python -m examples.graph.discord_workflow
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Add parent directory to path if needed
file = os.path.abspath(__file__)
parent = os.path.dirname(os.path.dirname(os.path.dirname(file)))
sys.path.insert(0, parent)

from alchemist.ai.graph.base import Graph, NodeState
from alchemist.ai.graph.nodes import AgentNode, TerminalNode
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.logging import (
    configure_logging,
    LogLevel,
    LogComponent,
    get_logger,
    Colors,
    VerbosityLevel,
    AlchemistLoggingConfig
)
from alchemist.extensions.discord.runtime import (
    DiscordRuntimeConfig,
    DiscordRuntime
)
from mirascope.core import prompt_template, Messages

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "discord_workflow.log"),
        logging.StreamHandler()
    ]
)

logger = get_logger(LogComponent.WORKFLOW)


@prompt_template()
def basic_response(messages: str, bot_id: str) -> Messages.Type:
    """
    Basic prompt template for responding to Discord messages.

    Args:
        messages (str): A combined string of relevant Discord messages.
        bot_id (str): The bot's Discord ID for mention checking.

    Returns:
        Messages.Type: The Mirascope Messages object with user prompt data.
    """
    return Messages.User(
        f"You were mentioned in this message. Review the conversation and provide a helpful response:\n\n{messages}"
    )


class MentionCheckNode(Node):
    """Node that checks if the bot was mentioned in the message."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Check if the message mentions the bot.
        
        Args:
            state: Current node state containing message data
            
        Returns:
            str: Next node ID based on mention check
        """
        message_data = state.data.get("discord_incoming", {})
        mentions = message_data.get("mentions", [])
        bot_id = state.data.get("bot_id")
        
        if str(bot_id) in mentions:
            logger.info(f"🤖 Bot mentioned in message")
            return "discord_agent"
        
        logger.debug("Bot not mentioned, skipping message")
        return "end"


async def create_discord_workflow() -> Graph:
    """
    Create and return a graph workflow for Discord-based interactions.
    This graph has one AgentNode that summarizes or reacts to messages,
    followed by a TerminalNode to indicate completion.

    Returns:
        Graph: The configured graph instance.
    """
    # Create the graph with an Alchemist logging config
    graph = Graph(
        logging_config=AlchemistLoggingConfig(
            level=VerbosityLevel.INFO,
            show_llm_messages=True,
            show_node_transitions=True,
            show_tool_calls=False
        )
    )

    # Add mention check node
    mention_check = MentionCheckNode(
        id="check_mention",
        next_nodes={
            "discord_agent": "discord_agent",
            "end": "end"
        }
    )

    # Agent node that uses the prompt to respond
    agent_node = AgentNode(
        id="discord_agent",
        prompt_template=basic_response,
        agent=BaseAgent(),
        input_map={
            "messages": "data.discord_incoming.content",
            "bot_id": "data.bot_id"
        },
        next_nodes={"default": "end"}
    )

    # Terminal node
    end_node = TerminalNode(id="end")

    # Add nodes to graph
    for node in [mention_check, agent_node, end_node]:
        graph.add_node(node)

    # Define the entry point
    graph.add_entry_point("start", "check_mention")

    return graph


async def main() -> None:
    """
    Initialize the Discord bot runtime, set up the workflow, and start the bot.
    The bot will periodically fetch new messages, run them through the Alchemist
    graph workflow, and post responses.
    """
    try:
        load_dotenv()
        token = os.environ.get("DISCORD_BOT_TOKEN", None)
        if not token:
            raise ValueError("No Discord token found. Set DISCORD_BOT_TOKEN in .env or env variables.")

        # Create the workflow
        workflow = await create_discord_workflow()

        # Configure the Discord runtime
        config = DiscordRuntimeConfig(
            provider="openpipe",
            model="gpt-4o-mini",  # Example model name
            # persona=AUG_E,
            platform_config={
                "workflow": workflow,
                "check_interval": 30,  # Check for messages every 30s
            },
            bot_token=token,
            # Example channel IDs; replace with real IDs if needed
            channel_ids=["1318659602115592204"]
        )

        # Create and start the Discord runtime
        runtime = DiscordRuntime(config=config)

        logger.info("Starting Discord workflow bot...")
        await runtime.start()

        # Keep the bot running indefinitely
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await runtime.stop()

    except Exception as e:
        logger.error(f"Error in Discord workflow: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())