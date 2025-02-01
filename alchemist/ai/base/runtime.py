"""Base Runtime Module for Agent Execution Environments"""

from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
import logging
import re
import asyncio

from alchemist.ai.base.logging import LogComponent
from alchemist.ai.base.agent import BaseAgent

# Get logger for runtime component
logger = logging.getLogger(LogComponent.RUNTIME.value)

class Session(BaseModel):
    """Tracks runtime session data."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: str
    start_time: datetime = Field(default_factory=datetime.now)
    agent_config: Dict[str, Any] = Field(default_factory=dict)

class RuntimeConfig(BaseModel):
    """Configuration for runtime environments.
    
    Attributes:
        provider: The LLM provider to use (default: "openai")
        model: The model to use (default: "gpt-4o-mini")
        system_prompt: System prompt configuration as string or Pydantic model
        tools: List of available tools
        platform_config: Additional platform-specific configuration
    """
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    system_prompt: Union[str, BaseModel]
    tools: list = Field(default_factory=list)
    platform_config: Dict[str, Any] = Field(default_factory=dict)

class BaseRuntime(ABC):
    """Abstract base for all runtime environments."""
    
    def __init__(self, agent: BaseAgent, config: Optional[RuntimeConfig] = None) -> None:
        """Initialize runtime with an agent instance.
        
        Args:
            agent: Instance of BaseAgent or its subclasses
            config: Optional runtime configuration
        """
        self.agent = agent
        self.config = config or RuntimeConfig(
            system_prompt=agent.system_prompt,
            tools=agent.tools
        )
        self.current_session = None

    @abstractmethod
    def _create_agent(self):
        """Create appropriate agent instance."""
        pass

    def _start_session(self, platform: str) -> None:
        self.current_session = Session(
            platform=platform,
            agent_config=self.config.model_dump()
        )
        
    @abstractmethod
    async def start(self) -> None:
        """Start the runtime."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the runtime."""
        pass

class BaseChatRuntime(BaseRuntime):
    """Base class for chat-based runtime environments."""
    
    def _create_agent(self):
        """Create agent instance."""
        from alchemist.ai.base.agent import BaseAgent
        return BaseAgent(
            system_prompt=self.config.system_prompt,  # Updated to use system_prompt
            tools=self.config.tools
        )
    
    async def process_message(self, message: str) -> str:
        """Process a single message and return the response."""
        if not self.current_session:
            self._start_session(platform="chat")
        try:
            response = await self.agent._step(message)
            logger.debug(f"Agent response: {response}")
            return response
        except Exception as e:
            logger.exception("Error processing message")
            raise

class LocalRuntime(BaseChatRuntime):
    """Runtime for local console chat interactions."""
    
    async def start(self) -> None:
        """Start a local chat session."""
        self._start_session("local")
        logger.info("Starting chat session. Type 'exit' or 'quit' to stop.")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                response = await self.process_message(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                logger.info("Chat session interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n[Error] {str(e)}")
        
        await self.stop()
        logger.info("Chat session ended. Goodbye! ✨")
    
    async def stop(self) -> None:
        """Stop the local runtime."""
        pass

__all__ = ["RuntimeConfig", "BaseRuntime", "BaseChatRuntime", "LocalRuntime"]
