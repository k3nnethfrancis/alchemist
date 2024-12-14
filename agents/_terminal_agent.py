"""
Previously working implementation of the Terminal agent that provides a natural language interface to various tools.
"""

from datetime import datetime
from typing import ClassVar, List, Type
from pydantic import BaseModel
from mirascope.core import (
    BaseMessageParam,
    Messages,
    openai,
    prompt_template,
    BaseTool
)
from mirascope.tools import DuckDuckGoSearch, ParseURLContent
from tools.twitter_client import CheckTwitterFeed, WriteTwitterTweet

class TerminalAgent(BaseModel):
    messages: list[BaseMessageParam] = []
    command_history: list[str] = []
    max_results_per_query: int = 2
    tools: ClassVar[List[Type[BaseTool]]] = [
        DuckDuckGoSearch, 
        ParseURLContent, 
        CheckTwitterFeed, 
        WriteTwitterTweet
    ]

    @openai.call(model="gpt-4o-mini", stream=True)
    @prompt_template(
        """
        SYSTEM:
        You are an expert terminal interface that helps users navigate their workflow using natural language.
        The current date is {current_date}.

        Your role is to:
        1. Understand the user's intent from their natural language input
        2. Help them accomplish their goals using the available tools
        3. Maintain context of the conversation and previous commands
        4. Think step by step about complex requests
        5. Be proactive in suggesting related actions that might help

        Previous command context: {self.command_history}
        Current user request: {question}

        You have access to the following tools:
        {self._get_tool_descriptions}

        When using tools:
        1. Consider the context of previous commands
        2. Break down complex requests into steps
        3. Explain what you're doing and why
        4. Confirm success or explain failures

        For Twitter operations:
        - Use check_twitter_feed to view recent tweets
        - Use write_tweet to post new tweets
        - Always confirm tweet posting success
        - Show tweet content after posting

        MESSAGES: {self.messages}
        USER: {question}
        """
    )
    def _stream(self, question: str) -> openai.OpenAIDynamicConfig:
        return {
            "tools": self.tools,
            "computed_fields": {
                "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
        }

    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        tool_descriptions = []
        for tool in self.tools:
            params = ""
            if hasattr(tool, 'parameters'):
                params = f" (parameters: {', '.join(tool.parameters.keys())})"
            tool_descriptions.append(f"- `{tool._name()}`: {tool.description}{params}")
        return "\n".join(tool_descriptions)

    def _step(self, question: str):
        response = self._stream(question)
        tools_and_outputs = []
        for chunk, tool in response:
            if tool:
                print(f"\nExecuting: {tool._name()} with parameters: {tool.args}")
                tools_and_outputs.append((tool, tool.call()))
            else:
                print(chunk.content, end="", flush=True)
        
        if response.user_message_param:
            self.messages.append(response.user_message_param)
        self.messages.append(response.message_param)
        
        if tools_and_outputs:
            self.messages += response.tool_message_params(tools_and_outputs)
            self._step("")

    def run(self):
        print("\nTerminal Agent initialized. Type 'exit' to quit.")
        print("Available tools:", ", ".join(tool._name() for tool in self.tools))
        print("How can I help you today?\n")
        
        while True:
            question = input("\n(User): ")
            if question.lower() == "exit":
                print("\nTerminal Agent shutting down. Goodbye!")
                break
            print("(Terminal): ", end="", flush=True)
            self._step(question)

if __name__ == "__main__":
    TerminalAgent().run()