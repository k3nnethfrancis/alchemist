"""Base Runtime Module for Agent Execution Environments"""

from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
import logging
import re
import asyncio
import discord

from alchemist.core.logger import log_step, log_run
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.core.extensions.discord.client import DiscordClient

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

class DiscordRuntime(BaseChatRuntime):
    """Runtime for Discord chat interactions.
    
    This runtime integrates with Discord's API to provide a chat interface
    through Discord servers and channels.
    
    Attributes:
        token (str): Discord bot token for authentication
        client (Optional[DiscordClient]): Instance of the Discord client
    """
    
    def __init__(self, config: RuntimeConfig, token: str):
        """Initialize Discord runtime.
        
        Args:
            config (RuntimeConfig): Configuration for the runtime
            token (str): Discord bot token for authentication
        """
        super().__init__(config)
        self.token = token
        self.client = None
        
    async def start(self) -> None:
        """Start Discord bot session.
        
        This method:
        1. Initializes a Discord session
        2. Sets up required intents
        3. Creates and starts the Discord client
        
        Raises:
            ValueError: If token is invalid
            ConnectionError: If connection to Discord fails
        """
        self._start_session("discord")
        
        # Setup Discord client with required intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        
        try:
            self.client = DiscordClient(
                agent=self.agent,
                intents=intents,
                token=self.token
            )
            
            logger.info("Starting Discord bot...")
            await self.client.start()
            
        except discord.errors.LoginFailure as e:
            logger.error(f"Failed to login to Discord: {str(e)}")
            raise ValueError("Invalid Discord token provided") from e
        except Exception as e:
            logger.error(f"Error starting Discord client: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop Discord bot session.
        
        Gracefully closes the Discord client connection.
        """
        if self.client:
            try:
                await self.client.close()
                logger.info("Discord bot stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Discord client: {str(e)}")
                raise

__all__ = ["RuntimeConfig", "BaseRuntime", "BaseChatRuntime", "LocalRuntime", "DiscordRuntime"]
