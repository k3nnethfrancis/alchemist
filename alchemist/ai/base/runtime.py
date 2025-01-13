"""Base Runtime Module for Agent Execution Environments"""

from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
import logging
import re
import asyncio

from alchemist.core.logger import log_step, log_run
from alchemist.ai.prompts.base import PersonaConfig

logger = logging.getLogger(__name__)

class Session(BaseModel):
    """Tracks runtime session data."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: str
    start_time: datetime = Field(default_factory=datetime.now)
    agent_config: Dict[str, Any] = Field(default_factory=dict)

class RuntimeConfig(BaseModel):
    """Configuration for runtime environments."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    persona: Union[Dict[str, Any], PersonaConfig]  # Accept either type
    tools: list = Field(default_factory=list)
    platform_config: Dict[str, Any] = Field(default_factory=dict)

    @property
    def persona_config(self) -> PersonaConfig:
        """Ensure persona is always a PersonaConfig object."""
        if isinstance(self.persona, dict):
            return PersonaConfig(**self.persona)
        return self.persona

class BaseRuntime(ABC):
    """Abstract base for all runtime environments."""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.agent = self._create_agent()
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
            provider=self.config.provider,
            model=self.config.model,
            persona=self.config.persona_config,
            tools=self.config.tools
        )
    
    async def process_message(self, message: str) -> str:
        """Process a single message and return the response."""
        if not self.current_session:
            self._start_session(platform="chat")
        # Handle both async and sync _step methods
        response = self.agent._step(message)
        if asyncio.iscoroutine(response):
            response = await response
        return response

class LocalRuntime(BaseChatRuntime):
    """Runtime for local console chat interactions."""
    
    async def start(self) -> None:
        """Start a local chat session."""
        self._start_session("local")
        print("\nStarting chat session. Type 'exit' or 'quit' to stop.")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                response = await self.process_message(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n[Error] {str(e)}")
        
        await self.stop()
        print("\nChat session ended. Goodbye! ✨")
    
    async def stop(self) -> None:
        """Stop the local runtime."""
        pass

__all__ = ["RuntimeConfig", "BaseRuntime", "BaseChatRuntime", "LocalRuntime"]
