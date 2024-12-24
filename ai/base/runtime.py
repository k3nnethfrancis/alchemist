"""
Base Runtime Module

This module defines the core BaseRuntime class that manages agent execution environments.
It provides:
- Platform-agnostic agent execution
- Step processing framework
- Session management
- Error handling and logging
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field

from ai.base.agent import BaseAgent

import logging
logger = logging.getLogger(__name__)

class Session(BaseModel):
    """
    Tracks runtime session data.
    
    Attributes:
        id (str): Unique session identifier
        platform (str): Platform this session is running on
        start_time (datetime): When the session started
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: str
    start_time: datetime = Field(default_factory=datetime.now)

class BaseRuntime(ABC):
    """
    Base class for runtime environments.
    
    This class provides:
    - Session management
    - Message processing
    - Error handling
    
    Attributes:
        agent (BaseAgent): The agent instance
        config (Dict): Runtime configuration
        current_session (Session): Current runtime session
    """
    
    def __init__(self, agent: BaseAgent, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the runtime.
        
        Args:
            agent (BaseAgent): The agent instance
            config (Optional[Dict]): Runtime configuration
        """
        self.agent = agent
        self.config = config or {}
        self.current_session = None
    
    def _start_session(self, platform: str) -> None:
        """
        Start a new runtime session.
        
        Args:
            platform (str): The platform this session is running on
        """
        self.current_session = Session(platform=platform)
        logger.info(f"Started {platform} session {self.current_session.id}")
    
    @abstractmethod
    async def start(self) -> None:
        """Start the runtime."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the runtime."""
        pass
    
    async def process_message(self, message: str) -> str:
        """
        Process a message through the agent.
        
        Args:
            message (str): The message to process
            
        Returns:
            str: The agent's response
        """
        try:
            return await self.agent._step(message)
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I apologize, but something went wrong: {str(e)}"
