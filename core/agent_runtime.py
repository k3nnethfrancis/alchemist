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
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field

from core.logger import log_session
from agents.chat.agent import ChatAgent
from extensions.discord_client import DiscordClient
# Future imports for other agents and extensions
# from agents.workflow.agent import WorkflowAgent
# from extensions.slack_client import SlackClient
# etc.

logger = logging.getLogger(__name__)

class RuntimeSession(BaseModel):
    """Track runtime session data."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    start_time: datetime = Field(default_factory=datetime.now)
    interface: str
    messages: list = Field(default_factory=list)

class RuntimeConfig(BaseModel):
    """Configuration for the runtime environment."""
    agent_type: Literal["chat"]  # Add more types as developed
    extension: Literal["discord"]  # Add more extensions as developed
    provider: Literal["openai", "anthropic"]
    model: str
    extension_config: Optional[dict[str, Any]] = None
    log_dir: str = Field(default="data/sessions")

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
        self.current_session: Optional[RuntimeSession] = None
        
    def _start_session(self, interface: str) -> None:
        """Start a new runtime session."""
        self.current_session = RuntimeSession(interface=interface)
        logger.info(f"Started new session {self.current_session.session_id}")

    def _end_session(self) -> None:
        """End current session and log it."""
        if self.current_session:
            log_path = Path(self.config.log_dir) / self.current_session.interface / f"{self.current_session.session_id}.json"
            log_session(
                session=self.current_session,
                agent=self.agent,
                log_path=log_path
            )
            self.current_session = None

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

    async def get_response(self, message: str) -> str:
        """Process a message through the configured agent."""
        if not self.current_session:
            self._start_session(self.config.extension)
            
        if not message.strip():
            return ""
        
        # Log incoming message
        if self.current_session:
            self.current_session.messages.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
        
        # Get response from agent - now properly awaited
        response = await self.agent._step(message)
        
        # Log response
        if self.current_session:
            self.current_session.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
        
        return response

    async def start(self) -> None:
        """Start the runtime with the configured extension."""
        try:
            if self.config.extension == "discord":
                await self.extension.start()
            else:
                raise ValueError(f"Unsupported extension: {self.config.extension}")
        finally:
            self._end_session()