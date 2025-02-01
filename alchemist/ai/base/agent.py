"""
BaseAgent rewritten to accept a system prompt, removing direct persona references.

This module provides a provider-agnostic base class for AI Agents using Mirascope.
It handles:
 - Session state
 - Tool usage
 - Asynchronous calls
 - Prompt composition (system + conversation + user)

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
from typing import List, Any, Optional, Union
from pydantic import BaseModel, Field
import inspect
# import lilypad
from mirascope.core import (
    BaseMessageParam,
    BaseTool,
    BaseDynamicConfig,
    openai,
    Messages,
    prompt_template,
)
from openpipe import OpenAI as OpenPipeClient
from alchemist.ai.tools.calculator import CalculatorTool
from alchemist.ai.base.logging import LogComponent, AlchemistLoggingConfig, log_verbose, VerbosityLevel

# Get logger for agent component
logger = logging.getLogger(LogComponent.AGENT.value)


@prompt_template()
def create_system_prompt(config: BaseModel) -> list[BaseMessageParam]:
    """Creates a formatted system prompt from a Pydantic model.
    
    Args:
        config: A Pydantic model containing system prompt configuration
        
    Returns:
        list[BaseMessageParam]: List containing the formatted system message
    """
    def format_value(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        elif isinstance(value, dict):
            return "\n".join(f"- {k}: {v}" for k, v in value.items())
        elif isinstance(value, BaseModel):
            return format_model(value)
        return str(value)

    def format_model(model: BaseModel) -> str:
        sections = []
        for field_name, field_value in model.model_dump().items():
            field_title = field_name.replace('_', ' ').title()
            formatted_value = format_value(field_value)
            sections.append(f"{field_title}: {formatted_value}")
        return "\n\n".join(sections)

    content = format_model(config)
    logger.debug(f"[create_system_prompt] Formatted content:\n{content}")
    
    return [BaseMessageParam(role="system", content=content)]

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
        system_prompt: System prompt or Pydantic model for agent configuration
        tools: List of available tool classes (not instances)
        logging_config: Logging configuration for controlling verbosity
    """
    
    history: List[BaseMessageParam] = Field(
        default_factory=list,
        description="Conversation history"
    )
    system_prompt: Optional[Union[str, BaseModel]] = None
    tools: List[type[BaseTool]] = Field(
        default_factory=list,
        description="List of tool classes available to the agent"
    )
    logging_config: AlchemistLoggingConfig = Field(
        default_factory=AlchemistLoggingConfig,
        description="Controls the verbosity and detail of agent logs."
    )

    # @lilypad.generation()
    @openai.call("gpt-4o-mini", client=OpenPipeClient())
    # @openai.call("gpt-4o-mini")
    def _call(self, query: str) -> BaseDynamicConfig:
        """Make an OpenPipe API call with the current conversation state.
        
        This method prepares the messages for the API call by:
        1. Creating a system message from the system_prompt
        2. Including the full conversation history
        3. Adding the current query if present
        4. Providing available tools to the model
        
        Args:
            query: The current user input, or empty string for follow-up calls
            
        Returns:
            BaseDynamicConfig: Contains messages and available tools
        """
        # Create system message based on type
        if isinstance(self.system_prompt, BaseModel):
            system_messages = create_system_prompt(self.system_prompt)
        else:
            system_messages = [BaseMessageParam(role="system", 
                                             content=self.system_prompt or "")]
            
        messages = [
            *system_messages,
            *self.history
        ]
        
        if query:
            messages.append(BaseMessageParam(role="user", content=query))
            
        return {
            "messages": messages,
            "tools": self.tools
        }

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
        
        Logging Details:
            - At VERBOSE or DEBUG levels, logs conversation messages and tool calls.
            - At INFO level, logs only high-level steps.
        
        Args:
            query: User input or empty string for follow-up steps
            
        Returns:
            str: The final response content
        """
        if query:
            self.history.append(BaseMessageParam(role="user", content=query))
            if self.logging_config.show_llm_messages or \
               self.logging_config.level <= VerbosityLevel.DEBUG:
                log_verbose(logger, f"Added user query to history: {query}")
            
        response = self._call(query)
        self.history.append(response.message_param)
        
        if self.logging_config.show_llm_messages or \
           self.logging_config.level <= VerbosityLevel.DEBUG:
            log_verbose(logger, f"Agent LLM response: {response.content}")
        
        tools_and_outputs = []
        if tools := response.tools:
            if self.logging_config.show_tool_calls or \
               self.logging_config.level <= VerbosityLevel.DEBUG:
                log_verbose(logger, f"Tools to call: {tools}")
            
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
                
            # Get the result first before printing anything
            result = await self._step(query)
            
            # Print the assistant's message all at once
            print(f"(Assistant): {result}")
# Main execution block for direct script usage
if __name__ == "__main__":
    import asyncio
    from alchemist.ai.base.logging import configure_logging, LogLevel
    
    async def main():
        """Run the agent."""
        configure_logging(
            default_level=LogLevel.INFO,
            component_levels={
                LogComponent.AGENT: LogLevel.DEBUG
            }
        )
        
        agent = BaseAgent(tools=[CalculatorTool])
        
        logger.info("\nInitialized agent with gpt-4o-mini")
        logger.info("Type 'exit' or 'quit' to end the conversation")
        logger.info("-" * 50)
        
        await agent.run()
    
    asyncio.run(main())