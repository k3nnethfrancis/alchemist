"""
Agent Runtime Module

This module provides a runtime environment for managing AI agents and their interactions
with different frontend interfaces (like Discord, CLI, etc.).

Key Features:
- Manages different types of agents (Chat, Workflow, Autonomous, etc.)
- Handles extension routing and configuration
- Provides a clean interface for different frontend extensions
- Supports multiple agent configurations
- Centralizes agent management logic

Usage:
    runtime = AgentRuntime(
        agent_type="chat",
        extension="discord",
        provider="anthropic",
        model="claude-3-5-sonnet-20240620"
    )
    await runtime.start()
"""

import logging
from typing import Literal, Optional, Any
from pydantic import BaseModel

from agents.chat.agent import ChatAgent
from extensions.discord_client import DiscordClient
# Future imports for other agents and extensions
# from agents.workflow.agent import WorkflowAgent
# from extensions.slack_client import SlackClient
# etc.

logger = logging.getLogger(__name__)

class RuntimeConfig(BaseModel):
    """Configuration for the runtime environment."""
    agent_type: Literal["chat"]  # Add more types as developed
    extension: Literal["discord"]  # Add more extensions as developed
    provider: Literal["openai", "anthropic"]
    model: str
    extension_config: Optional[dict[str, Any]] = None

class AgentRuntime:
    """
    A runtime environment for managing different types of agents and their extensions.
    
    This class serves as the primary orchestrator for:
    1. Agent initialization and configuration
    2. Extension setup and routing
    3. Runtime lifecycle management
    """

    def __init__(self, config: RuntimeConfig):
        """
        Initialize the runtime with specified agent and extension configurations.

        Args:
            config (RuntimeConfig): Configuration specifying agent type, extension,
                                  and other runtime parameters.
        """
        self.config = config
        self.agent = self._initialize_agent()
        self.extension = self._initialize_extension()

    def _initialize_agent(self) -> Any:
        """Initialize the appropriate agent based on configuration."""
        if self.config.agent_type == "chat":
            return ChatAgent(
                provider=self.config.provider
            )
        # Add more agent types as they're developed
        raise ValueError(f"Unsupported agent type: {self.config.agent_type}")

    def _initialize_extension(self) -> Any:
        """Initialize the appropriate extension based on configuration."""
        if self.config.extension == "discord":
            return DiscordClient(
                agent_runtime=self,
                **self.config.extension_config
            )
        raise ValueError(f"Unsupported extension: {self.config.extension}")

    def get_response(self, message: str) -> str:
        """Process a message through the configured agent."""
        if not message.strip():
            return ""
        
        logger.debug("AgentRuntime received message: '%s'", message)
        
        # Get response from agent
        response = self.agent._step(message)
        
        # Log the agent's history to debug tool usage
        for msg in self.agent.history[-3:]:  # Last 3 messages
            logger.debug("History entry: %s", msg)
        
        # Log any generated images
        if self.agent.generated_images:
            logger.debug("Generated images: %s", self.agent.generated_images)
        
        return response

    async def start(self) -> None:
        """Start the runtime with the configured extension."""
        if self.config.extension == "discord":
            await self.extension.start()
        else:
            raise ValueError(f"Unsupported extension: {self.config.extension}")