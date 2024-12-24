"""Base Agent Module

This module defines the core BaseAgent class that all agent implementations inherit from.
It provides the foundational structure for:
- LLM provider integration via Mirascope
- Message history management
- Step-based processing framework
- Tool integration
"""

from typing import Optional, Union, Any, List
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    anthropic,
    openai
)
from core.mirascope.providers.openpipe import openpipe_call, OpenPipeMessageParam

import logging
logger = logging.getLogger(__name__)

class BaseAgent(BaseModel, ABC):
    """
    Abstract base class for all agent implementations.
    
    This class defines the core interface and shared functionality that all
    agents must implement. It provides:
    - Provider-agnostic LLM integration
    - Message history management
    - Abstract step interface
    - Tool registration and execution
    
    Attributes:
        history (list): List of conversation history entries
        provider (str): The LLM provider to use ('openai', 'anthropic', or 'openpipe')
        tools (list): List of available tools for the agent
    """
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    history: List[Union[openai.OpenAIMessageParam, anthropic.AnthropicMessageParam, OpenPipeMessageParam]] = Field(default_factory=list)
    provider: str = Field(default="openai")
    tools: List[Any] = Field(default_factory=list)
    
    @abstractmethod
    async def _step(self, message: str) -> str:
        """
        Process a single conversation step.
        
        This is the core method that must be implemented by each agent type.
        It defines how the agent processes each interaction step.
        
        Args:
            message (str): The input message to process
            
        Returns:
            str: The agent's response
        """
        pass
        
    @openpipe_call("gpt-4o-mini")
    def _call_openpipe(self, query: str) -> BaseDynamicConfig:
        """Process query using OpenPipe's model."""
        return {
            "messages": self.history,
            "tools": self.tools
        }

    @anthropic.call("claude-3-5-sonnet-20241022")
    def _call_anthropic(self, query: str) -> BaseDynamicConfig:
        """Process query using Anthropic's Claude model."""
        return {
            "messages": self.history,
            "tools": self.tools
        }

    @openai.call("gpt-4o-mini")
    def _call_openai(self, query: str) -> BaseDynamicConfig:
        """Process query using OpenAI's GPT-4 model."""
        return {
            "messages": self.history,
            "tools": self.tools
        }

    def add_tool(self, tool: Any) -> None:
        """
        Register a new tool with the agent.
        
        Args:
            tool: The tool to register
        """
        if tool not in self.tools:
            self.tools.append(tool)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role (str): The role of the message sender ('user' or 'assistant')
            content (str): The message content
        """
        message = BaseMessageParam(role=role, content=content)
        self.history.append(message)

    async def get_response(self, query: str) -> str:
        """
        Get a response for a query using the appropriate provider.
        
        This is a convenience method that wraps _step() and handles
        provider selection and error handling.
        
        Args:
            query (str): The query to process
            
        Returns:
            str: The agent's response
        """
        try:
            return await self._step(query)
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return f"An error occurred: {str(e)}"
