"""
Chat Agent Module

This module implements a ChatAgent that provides conversational interactions.
It builds on the BaseAgent class and adds chat-specific functionality.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from mirascope.core import BaseDynamicConfig, BaseMessageParam

from ai.base.agent import BaseAgent
from ai.prompts.base import create_system_prompt
from ai.prompts.persona import Persona, AUG_E
from ai.agents.chat.tools import ImageGenerationTool

import logging
logger = logging.getLogger(__name__)

class ChatAgent(BaseAgent):
    """
    A conversational agent that provides natural chat interactions.
    
    This agent is designed for direct conversations, with support for:
    - Natural language interactions
    - Persona-based responses
    - Tool usage when needed
    - Message history tracking
    
    Attributes:
        persona (Persona): The agent's personality configuration
    """
    
    persona: Persona
    
    def __init__(self, provider: str = "openai", persona: Optional[Persona] = None):
        """
        Initialize the chat agent.
        
        Args:
            provider (str): The LLM provider to use
            persona (Optional[Persona]): The agent's personality. Defaults to AUG_E
        """
        # Initialize with all required fields
        super().__init__(
            provider=provider,
            persona=persona or AUG_E
        )
        
        # Add default tools
        self.tools = [ImageGenerationTool]
        
        # Set up initial system prompt
        system_prompt = create_system_prompt(self.persona)
        self.add_message(role="system", content=system_prompt["content"])

    async def _step(self, message: str) -> str:
        """
        Process a single conversation step.
        
        This implementation:
        1. Adds user message to history
        2. Gets response from appropriate provider
        3. Handles any tool usage generically
        4. Returns the final response
        
        Args:
            message (str): The input message to process
            
        Returns:
            str: The agent's response
        """
        try:
            # Add user message to history
            if message:
                self.history.append(
                    BaseMessageParam(role="user", content=message)
                )
            
            # Get response from appropriate provider
            response = {
                "openai": self._call_openai,
                "anthropic": self._call_anthropic,
                "openpipe": self._call_openpipe
            }[self.provider](message)
            
            # Add response to history
            self.history.append(response.message_param)
            
            # Handle tools if present
            if tool := response.tool:
                logger.info(f"Calling tool '{tool.__class__.__name__}' with args {tool.model_dump()}")
                try:
                    # Call tool and get result
                    result = await tool.call()
                    
                    # Add tool response to history
                    self.history.extend(
                        response.tool_message_params([(tool, str(result))])
                    )
                    
                    # Get follow-up response after tool use
                    followup = await self._step("")
                    
                    # Let the tool format its own result in the response if needed
                    return tool.format_response(followup, result)
                    
                except Exception as e:
                    # Add error response to history
                    error_msg = str(e)
                    self.history.append(
                        BaseMessageParam(
                            role="assistant",
                            content=f"I apologize, but I encountered an error: {error_msg}"
                        )
                    )
                    return f"I apologize, but I encountered an error: {error_msg}"
            
            return response.content

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Step error: {error_msg}")
            
            # Add error response to history
            self.history.append(
                BaseMessageParam(
                    role="assistant", 
                    content=f"I apologize, but something went wrong: {error_msg}"
                )
            )
            return f"I apologize, but something went wrong: {error_msg}"
