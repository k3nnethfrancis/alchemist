"""
Chat agent module that handles conversations using multiple LLM providers.

This module implements a ChatAgent class that processes queries using both OpenAI, 
Anthropic, and OpenPipe models, maintains conversation history, and executes tools.
"""

from typing import Optional, Union, Any
from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    anthropic,
    openai
)
from core.providers.openpipe import openpipe_call, OpenPipeMessageParam
from pydantic import BaseModel, Field

from agents.chat.tools import ImageGenerationTool
from agents.chat.prompts.base import create_system_prompt, PersonaConfig
from agents.chat.prompts.persona import AUG_E
from core.logger import log_run

import logging
logger = logging.getLogger(__name__)

class ChatAgent(BaseModel):
    """
    A chat agent that processes queries using multiple LLM providers.
    
    Attributes:
        history (list): List of conversation history entries
        provider (str): The LLM provider to use ('openai', 'anthropic', or 'openpipe')
    """
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    history: list[Union[openai.OpenAIMessageParam, anthropic.AnthropicMessageParam, OpenPipeMessageParam]] = []
    provider: str = Field(default="openai")

    def __init__(self, provider: str = "openai", **data):
        super().__init__(provider=provider, **data)
        persona = PersonaConfig(**AUG_E)
        system_prompt = BaseMessageParam(**create_system_prompt(persona))
        self.history.append(system_prompt)

    @openpipe_call("gpt-4o-mini")
    def _call_openpipe(self, query: str) -> BaseDynamicConfig:
        """Process query using OpenPipe's model."""
        messages = [*self.history]
        if query:
            messages.append(BaseMessageParam(role="user", content=query))
        return {
            "messages": messages,
            "tools": [ImageGenerationTool]
        }

    @anthropic.call("claude-3-5-sonnet-20241022")
    def _call_anthropic(self, query: str) -> BaseDynamicConfig:
        """Process query using Anthropic's Claude model."""
        messages = [*self.history]
        if query:
            messages.append(BaseMessageParam(role="user", content=query))
        return {
            "messages": messages,
            "tools": [ImageGenerationTool]
        }

    @openai.call("gpt-4o-mini")
    def _call_openai(self, query: str) -> BaseDynamicConfig:
        """Process query using OpenAI's GPT-4 model."""
        messages = [*self.history]
        if query:
            messages.append(BaseMessageParam(role="user", content=query))
        return {
            "messages": messages,
            "tools": [ImageGenerationTool]
        }

    async def _step(self, query: str) -> str:
        """Process a single conversation step."""
        try:
            if query:
                self.history.append(
                    BaseMessageParam(role="user", content=query)
                )

            # Get response from appropriate provider
            response = {
                "openai": self._call_openai,
                "anthropic": self._call_anthropic,
                "openpipe": self._call_openpipe
            }[self.provider](query)

            # Add response to history
            self.history.append(response.message_param)
            
            # Handle tools if present
            if tool := response.tool:
                logger.info(f"Calling tool '{tool._name()}' with args {tool.args}")
                result = await tool.call()
                
                # Add tool response to history
                self.history.extend(
                    response.tool_message_params([(tool, str(result))])
                )
                # Get follow-up response
                return await self._step("")
            
            return response.content

        except Exception as e:
            logger.error(f"Step error: {str(e)}")
            return f"An error occurred: {str(e)}"

    @log_run(log_dir="logs/chat")
    async def run(self) -> None:
        """Run the chat agent in an interactive loop."""
        while True:
            query = input("User: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            print("Assistant: ", end="", flush=True)
            try:
                response = await self._step(query)
                print(response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def get_response(self, query: str) -> str:
        """Get response for a query."""
        return await self._step(query)

if __name__ == "__main__":
    import asyncio
    agent = ChatAgent(provider="openai")
    asyncio.run(agent.run())