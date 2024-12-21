"""
Chat agent module that handles conversations using multiple LLM providers.

This module implements a ChatAgent class that processes queries using both OpenAI and 
Anthropic models, maintains conversation history, and executes tools like image generation.
"""

from typing import Optional, Union

from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    anthropic,
    openai
)
from pydantic import BaseModel, Field

# import lilypad

from agents.chat.tools import ImageGenerator
from agents.chat.prompts.base import create_system_prompt, PersonaConfig
from agents.chat.prompts.persona import AUG_E

import logging

logger = logging.getLogger(__name__)


class ChatAgent(BaseModel):
    """
    A chat agent that processes queries using multiple LLM providers.
    
    Attributes:
        history (list): List of conversation history entries
        provider (str): The LLM provider to use ('openai' or 'anthropic')
        generated_images (list): Store generated image data
        persona (PersonaConfig): Configuration for the agent's persona
    """
    
    history: list[Union[openai.OpenAIMessageParam, anthropic.AnthropicMessageParam]] = []
    provider: str = Field(default="anthropic")
    generated_images: list[dict] = Field(default_factory=list)
    
    def __init__(self, provider: str = "anthropic", **data):
        # Create persona config first
        persona = PersonaConfig(**AUG_E)
        
        # Initialize base model
        super().__init__(provider=provider, **data)
        
        # Initialize system prompt with persona
        system_prompt = BaseMessageParam(**create_system_prompt(persona))
        self.history.append(system_prompt)

    # @lilypad.generation()
    @anthropic.call("claude-3-5-sonnet-20241022", tools=[ImageGenerator])
    def _call_anthropic(self, query: str) -> BaseDynamicConfig:
        """Process query using Anthropic's Claude model."""
        messages = [
            *self.history
        ]
        if query:
            messages.append(BaseMessageParam(role="user", content=query))
        return {"messages": messages}

    # @lilypad.generation()
    @openai.call("gpt-4o-mini", tools=[ImageGenerator])
    def _call_openai(self, query: str) -> BaseDynamicConfig:
        """Process query using OpenAI's GPT-4 model."""
        messages = [
            *self.history
        ]
        if query:
            messages.append(BaseMessageParam(role="user", content=query))
        return {"messages": messages}

    def _process_image_generation(self, tool_output: dict) -> str:
        """
        Process the output from the image generation tool.
        
        Args:
            tool_output: Dictionary containing image generation results
            
        Returns:
            str: Formatted response about the image generation
        """
        if tool_output["status"] == "success":
            self.generated_images.append(tool_output)
            return (
                f"I've generated an image based on your prompt: '{tool_output['prompt']}'\n"
                f"You can view it here: {tool_output['url']}\n\n"
                "Would you like me to generate another image or modify this one?"
            )
        else:
            return f"Image generation failed: {tool_output['message']}"

    def _step(self, query: str) -> str:
        """
        Process a single conversation step.
        
        Args:
            query: The user's input query
            
        Returns:
            str: The agent's response
        """
        # Add user query to history if it exists
        if query:
            self.history.append(BaseMessageParam(role="user", content=query))
        
        # Get LLM response
        response = self._call_anthropic(query) if self.provider == "anthropic" else self._call_openai(query)
        
        # Add assistant response to history
        self.history.append(response.message_param)
        
        # Handle tools if present
        tools_and_outputs = []
        if tool := response.tool:
            logger.info(f"[Calling Tool '{tool.__class__.__name__}' with args {tool.model_dump()}]")
            tool_output = tool.call()
            
            # For image generation, store the result
            if isinstance(tool, ImageGenerator):
                self._process_image_generation(tool_output)
                
            tools_and_outputs.append((tool, tool_output))
            self.history += response.tool_message_params(tools_and_outputs)
            return self._step("")  # Recursive call with empty query to continue conversation
        
        return response.content

    def run(self) -> None:
        """Run the chat agent in an interactive loop."""
        while True:
            query = input("User: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            print("Assistant: ", end="", flush=True)
            try:
                response = self._step(query)
                print(response)
            except Exception as e:
                print(f"\nError: {str(e)}")


if __name__ == "__main__":
    agent = ChatAgent(provider="openai")
    agent.run()
    ### tool test