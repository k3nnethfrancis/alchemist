"""
BaseAgent implementation for the Alchemist system.

This module defines a provider-agnostic agent that can:
- Send queries to OpenAI, Anthropic, or OpenPipe via Mirascope.
- Incorporate a PersonaConfig for a consistent "voice" and context.
- Maintain conversation history for multi-turn interactions.
- Operate in interactive mode (e.g., local console chatbot).
- Integrate seamlessly with the Graph system for workflow-based execution.

Core Features:
1. Multi-provider support:
   - "openai", "anthropic", or "openpipe" (via OpenPipeClient).
2. Persona-based context:
   - Uses alchemist.ai.prompts.base.PersonaConfig to represent persona attributes.
3. History tracking:
   - Maintains an ordered list of conversation turns.
4. Simple, modular design:
   - Encourages extension for specialized nodes or workflows.

Usage Examples:
----------------
1) Local Chat Example:
   from alchemist.ai.prompts.persona import KEN_E
   from alchemist.ai.prompts.base import PersonaConfig
   agent = BaseAgent(
       provider="openai",
       model="gpt-4o-mini",
       persona=PersonaConfig(**KEN_E)
   )
   agent.run()  # Starts a local interactive chat in the terminal.

2) Graph Integration:
   - Use in LLMNode or DecisionNode in the Graph system:
       llm_node = LLMNode(
           id="my_llm_step",
           agent=agent,
           prompt="Hello from {context_data}!"
       )
   - The node calls agent.get_response(...) or agent._step(...),
     enabling broader, multi-node workflow orchestration.

Contributing Guidelines:
------------------------
- Follow PEP 8 and add docstrings and type hints.
- Keep code modular and easy to extend.
- Include tests that verify both functionality and workflow integrity.
- Maintain minimal complexity and deep modules with clear interfaces.
"""

import sys
import logging
from typing import Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field

# Mirascope
from mirascope.core import (
    openai,
    anthropic,
    Messages,
    BaseMessageParam,
    prompt_template,
)
from mirascope.core.base import BaseCallResponse

# OpenPipe
from openpipe import OpenAI as OpenPipeClient

# Alchemist imports
from alchemist.ai.prompts.base import create_system_prompt, PersonaConfig
from alchemist.ai.prompts.persona import KEN_E  # Example persona dictionary

logger = logging.getLogger(__name__)


class BaseAgent(BaseModel):
    """
    Provider-agnostic chat agent for Alchemist.

    Capabilities:
    1) Multi-provider (OpenAI, Anthropic, OpenPipe) 
    2) Supports a PersonaConfig for consistent system instruction
    3) Maintains conversation history
    4) Usable alone (local chatbot) or in graph-based workflows
    """

    provider: Literal["openai", "anthropic", "openpipe"] = Field(
        default="openai",
        description="LLM provider: 'openai', 'anthropic', or 'openpipe'."
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model name, e.g. 'gpt-4o-mini', 'claude-3-5-sonnet-20240620', or an OpenPipe ID."
    )
    persona: PersonaConfig = Field(
        default_factory=PersonaConfig,
        description="Persona configuration, defines the agent's 'character.'"
    )
    history: List[BaseMessageParam] = Field(
        default_factory=list,
        description="Conversation history (system, user, assistant messages)."
    )

    class Config:
        arbitrary_types_allowed = True

    @prompt_template()
    def _system_prompt(self) -> Messages.Type:
        """
        Renders the system prompt using the persona config,
        ensuring consistent 'character' for this agent.
        """
        sys_msg_dict = create_system_prompt(self.persona)
        return [Messages.System(sys_msg_dict["content"])]

    def _call_llm(self, user_input: str) -> BaseCallResponse:
        """
        Generic interface for calling Mirascope-based LLMs.

        Args:
            user_input (str): User query or message.

        Returns:
            BaseCallResponse: The LLM response (containing text content, tokens, etc.).
        """
        # Build an in-memory prompt function that merges system prompt + history + user input
        @prompt_template()
        def messages_prompt(agent_input: str) -> Messages.Type:
            return [
                # The system prompt (from persona)
                *self._system_prompt(),
                # Past conversation history
                *self.history,
                # New user query
                Messages.User(agent_input),
            ]

        # Choose the method of invocation based on self.provider
        if self.provider == "openai":
            llm_fn = openai.call(self.model)(messages_prompt)
        elif self.provider == "anthropic":
            llm_fn = anthropic.call(self.model)(messages_prompt)
        elif self.provider == "openpipe":
            # Re-use the openai call pipeline but pass a different client
            llm_fn = openai.call(self.model, client=OpenPipeClient())(messages_prompt)
        else:
            raise ValueError(f"Unrecognized provider '{self.provider}'")

        response = llm_fn(user_input)
        return response

    def get_response(self, user_input: str) -> str:
        """
        Obtain a single response string from the LLM without mutating self.history.

        Args:
            user_input (str): The user's query.

        Returns:
            str: The assistant's response content.
        """
        logger.debug(f"get_response called with query: {user_input}")
        response = self._call_llm(user_input)
        return response.content

    def _step(self, user_input: str) -> str:
        """
        Obtain a single response from the LLM and update the internal history.

        Args:
            user_input (str): The user's query.

        Returns:
            str: The assistant's response content (also appended to conversation history).
        """
        logger.debug(f"_step called with query: {user_input}")
        response = self._call_llm(user_input)
        assistant_msg = response.message_param

        # Update conversation history
        self.history.append(Messages.User(user_input))
        self.history.append(assistant_msg)

        return response.content

    def run(self) -> None:
        """
        Run an interactive local chat session in the console.
        Exits on 'exit' or 'quit'.

        Example usage:
            agent = BaseAgent(provider="openai", model="gpt-4o-mini")
            agent.run()
        """
        print("\nStarting interactive chat. Type 'exit' or 'quit' to stop.")
        while True:
            user_query = input("(User): ")
            if user_query.strip().lower() in ["exit", "quit"]:
                break
            try:
                answer = self._step(user_query)
                print(f"(Assistant): {answer}")
            except Exception as e:
                logger.error(f"Error in interactive loop: {e}")
                print(f"[Error] {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.info("Running BaseAgent as a standalone local chatbot...")

    # Use a sample persona from persona.py
    agent = BaseAgent(
        provider="openpipe",
        model="gpt-4o-mini",
        persona=PersonaConfig(**KEN_E)
    )
    agent.run()