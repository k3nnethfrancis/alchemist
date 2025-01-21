"""Base Agent implementation using Mirascope's agent patterns.

This module implements a flexible agent architecture using Mirascope's decorator pattern.
It uses OpenPipe for LLM calls, supports tool integration, and maintains conversation history.

Message Flow:
    1. User input is received and added to history
    2. OpenPipe API call is made with:
       - System message (from persona)
       - Conversation history
       - Current user query
    3. Response is processed:
       - If no tools: response is added to history and returned
       - If tools: each tool is executed, results added to history, and step repeats
    
Key Features:
    - OpenPipe integration with gpt-4o-mini
    - Tool integration with Mirascope's BaseTool pattern
    - Conversation history management
    - Persona-based behavior configuration
    - Async-first implementation

Example:
    ```python
    from alchemist.ai.base.tools import CalculatorTool
    
    agent = BaseAgent(tools=[CalculatorTool])
    await agent.run()
    ```
"""

import logging
from typing import List, Any
from pydantic import BaseModel, Field
import inspect

from mirascope.core import (
    BaseMessageParam,
    BaseTool,
    BaseDynamicConfig,
    openai,
)
from openpipe import OpenAI as OpenPipeClient

from alchemist.ai.prompts.base import create_system_prompt, PersonaConfig
from alchemist.ai.prompts.persona import BASE_ASSISTANT
from alchemist.ai.tools.calculator import CalculatorTool

logger = logging.getLogger(__name__)

class BaseAgent(BaseModel):
    """Base agent class implementing core agent functionality with persona support and tools.
    
    The agent maintains conversation history and supports tool execution through Mirascope's
    BaseTool pattern. It uses OpenPipe's gpt-4o-mini model for generating responses and
    deciding when to use tools.
    
    Message Flow:
        1. User messages are added to history
        2. System prompt (from persona) and history are sent to OpenPipe
        3. If the response includes tool calls:
           - Tools are executed in sequence
           - Results are added to history
           - Another API call is made with the tool results
        4. Final response is returned to the user
    
    Attributes:
        history: List of conversation messages (BaseMessageParam)
        persona: Configuration for agent's personality and behavior
        tools: List of available tool classes (not instances)
    """
    
    history: List[BaseMessageParam] = Field(
        default_factory=list,
        description="Conversation history"
    )
    persona: PersonaConfig = Field(
        default_factory=lambda: PersonaConfig(**BASE_ASSISTANT),
        description="Persona configuration for the agent"
    )
    tools: List[type[BaseTool]] = Field(
        default_factory=list,
        description="List of tool classes available to the agent"
    )

    @openai.call("gpt-4o-mini", client=OpenPipeClient())
    def _call(self, query: str) -> BaseDynamicConfig:
        """Make an OpenPipe API call with the current conversation state.
        
        This method prepares the messages for the API call by:
        1. Creating a system message from the persona
        2. Including the full conversation history
        3. Adding the current query if present
        4. Providing available tools to the model
        
        Args:
            query: The current user input, or empty string for follow-up calls
            
        Returns:
            BaseDynamicConfig: Contains messages and available tools
        """
        messages = [
            BaseMessageParam(role="system", content=create_system_prompt(self.persona)),
            *self.history,
            BaseMessageParam(role="user", content=query) if query else None,
        ]
        messages = [m for m in messages if m is not None]
        return {"messages": messages, "tools": self.tools}

    async def _step(self, query: str) -> str:
        """Execute a single step of agent interaction.
        
        This method handles the core interaction loop:
        1. Adds user query to history (if present)
        2. Makes API call and adds response to history
        3. If tools are in the response:
           - Executes each tool
           - Adds results to history
           - Makes another API call (recursive step)
        4. Returns final response content
        
        The flow ensures that:
        - All messages are properly added to history
        - Tools are executed in the correct order
        - Results are formatted correctly
        - Recursive steps handle follow-up responses
        
        Args:
            query: User input or empty string for follow-up steps
            
        Returns:
            str: The final response content
        """
        if query:
            self.history.append(BaseMessageParam(role="user", content=query))
            
        response = self._call(query)
        self.history.append(response.message_param)
        
        tools_and_outputs = []
        if tools := response.tools:
            for tool in tools:
                logger.info(f"[Calling Tool '{tool._name()}' with args {tool.args}]")
                
                # Handle both sync and async tool calls
                if inspect.iscoroutinefunction(tool.call):
                    result = await tool.call()
                else:
                    result = tool.call()
                    
                logger.info(f"Tool result: {result}")
                tools_and_outputs.append((tool, result))
            
            self.history.extend(response.tool_message_params(tools_and_outputs))
            return await self._step("")
        else:
            return response.content

    async def run(self) -> None:
        """Run the agent interaction loop.
        
        This method:
        1. Prompts for user input
        2. Processes each query through _step()
        3. Prints responses
        4. Continues until user exits
        
        The loop handles:
        - User input collection
        - Exit commands ('exit' or 'quit')
        - Async execution of steps
        - Response display
        """
        while True:
            query = input("(User): ")
            if query.lower() in ["exit", "quit"]:
                break
            print("(Assistant): ", end="", flush=True)
            result = await self._step(query)
            print(result)

# Main execution block for direct script usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Run the agent."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        
        agent = BaseAgent(tools=[CalculatorTool])  # Pass tool class, not instance
        
        print("\nInitialized agent with gpt-4o-mini")
        print("Type 'exit' or 'quit' to end the conversation")
        print("-" * 50)
        
        await agent.run()
    
    asyncio.run(main())