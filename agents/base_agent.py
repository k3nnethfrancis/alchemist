"""Base agent class that standardizes message handling and tool integration."""

from typing import List, Optional, Tuple, Type, ClassVar
from datetime import datetime
from pydantic import BaseModel
from mirascope.core import (
    BaseMessageParam,
    Messages,
    openai,
    BaseDynamicConfig,
    BaseTool,
    prompt_template
)

class BaseAgent(BaseModel):
    """Base agent with standardized message handling and tool integration."""
    
    messages: List[BaseMessageParam] = []
    tools: ClassVar[List[Type[BaseTool]]] = []
    system_prompt: str = ""
    
    def _convert_to_message_param(self, message: dict | BaseMessageParam) -> BaseMessageParam:
        """Convert a message dict to BaseMessageParam if needed."""
        if isinstance(message, dict):
            role = message["role"]
            content = message["content"]
            if role == "system":
                return Messages.System(content)
            elif role == "user":
                return Messages.User(content)
            elif role == "assistant":
                return Messages.Assistant(content)
        return message

    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        tool_descriptions = []
        for tool in self.tools:
            params = ""
            if hasattr(tool, 'parameters'):
                params = f" (parameters: {', '.join(tool.parameters.keys())})"
            tool_descriptions.append(f"- `{tool._name()}`: {tool.description}{params}")
        return "\n".join(tool_descriptions)

    def _step(self, query: str) -> None:
        """Process a single conversation turn."""
        if query:
            self.messages.append(Messages.User(query))
            
        stream = self._stream(query)
        tools_and_outputs = []
        
        for chunk, tool in stream:
            if tool:
                print(f"\nExecuting: {tool._name()} with parameters: {tool.args}")
                tools_and_outputs.append((tool, tool.call()))
            else:
                print(chunk.content, end="", flush=True)
        
        self.messages.append(stream.message_param)
        
        if tools_and_outputs:
            self.messages += stream.tool_message_params(tools_and_outputs)
            self._step("")

    def get_last_message(self) -> BaseMessageParam:
        """Get the last message in a standardized format."""
        if not self.messages:
            return Messages.System("")
        return self._convert_to_message_param(self.messages[-1])

    @openai.call(model="gpt-4o-mini", stream=True)
    @prompt_template("""
        SYSTEM:
        {self.system_prompt}

        MESSAGES: {self.messages}
        USER: {query}
    """)
    def _stream(self, query: str) -> BaseDynamicConfig:
        """Default stream implementation that can be overridden."""
        return {
            "tools": self.tools,
            "computed_fields": {
                "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

    def run(self) -> None:
        """Run the agent in interactive mode."""
        print(f"\n{self.__class__.__name__} initialized. Type 'exit' to quit.")
        if self.tools:
            print("Available tools:", ", ".join(tool._name() for tool in self.tools))
        print("How can I help you today?\n")
        
        while True:
            query = input("\n(User): ")
            if query.lower() == "exit":
                print(f"\n{self.__class__.__name__} shutting down. Goodbye!")
                break
            print(f"({self.__class__.__name__}): ", end="", flush=True)
            self._step(query)