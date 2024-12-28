"""Base Runtime Module for Agent Execution Environments"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
import logging
import re

from alchemist.core.logger import log_step, log_run

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
    persona: Dict[str, Any]
    tools: list = Field(default_factory=list)
    platform_config: Dict[str, Any] = Field(default_factory=dict)

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
    """Base class for I/O chat operations."""
    
    async def process_message(self, content: str) -> str:
        """Process a chat message."""
        return await self.agent._step(content)

class LocalRuntime(BaseChatRuntime):
    """Runtime for local console chat interactions."""
    
    def _create_agent(self):
        """Create agent with local-specific configuration."""
        from alchemist.ai.base.agent import BaseAgent
        return BaseAgent(
            provider=self.config.provider,
            persona=self.config.persona,
            tools=self.config.tools
        )
    
    def _format_message(self, message: str, prefix: str = "") -> str:
        """Format message with proper line breaks and markdown-style links."""
        # Handle image markdown links
        message = re.sub(
            r'!\[(.*?)\]\((.*?)\)',
            lambda m: f"\n🖼️ Generated Image: {m.group(1)}\n{m.group(2)}\n",
            message
        )
        
        # Add prefix to first line
        lines = message.split('\n')
        if lines:
            lines[0] = f"{prefix}{lines[0]}"
            
        # Add proper indentation to subsequent lines
        indent = " " * len(prefix)
        for i in range(1, len(lines)):
            if lines[i].strip():
                lines[i] = f"{indent}{lines[i]}"
                
        return "\n".join(lines)
    
    @log_step(log_dir="logs/chat")
    async def process_message(self, content: str) -> str:
        """Process and log a chat message."""
        return await self.agent._step(content)
    
    async def start(self) -> None:
        """Start local chat session."""
        self._start_session("local")
        logger.info("Starting local chat session...")
        
        # Print welcome message
        print("\n" + "="*50)
        print(f"Starting chat with {self.config.persona.get('name', 'Assistant')}")
        print("Type 'exit' or 'quit' to end the session")
        print("="*50 + "\n")
        
        while True:
            # Get user input
            try:
                query = input("\n\033[94mYou:\033[0m ")  # Blue color for user
                if query.lower() in ["exit", "quit"]:
                    break
                
                # Process and format response
                try:
                    response = await self.process_message(query)
                    formatted_response = self._format_message(
                        response,
                        prefix="\033[92mAssistant:\033[0m "  # Green color for assistant
                    )
                    print(formatted_response)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    print(f"\n\033[91mError:\033[0m {str(e)}")  # Red color for errors
                    
            except KeyboardInterrupt:
                print("\n\nExiting chat session...")
                break
                
        await self.stop()
    
    async def stop(self) -> None:
        """Stop local chat session."""
        logger.info("Stopping local chat session...")
        print("\n" + "="*50)
        print("Chat session ended")
        print("="*50 + "\n")
