"""
DEMO FILE: NOT CURRENTLY IN USE

Terminal agent that provides a natural language interface to various tools.
"""

import asyncio
from typing import List, Type, ClassVar
from datetime import datetime
from mirascope.core import (
    BaseDynamicConfig,
    BaseTool,
    prompt_template
)
from mirascope.tools import DuckDuckGoSearch, ParseURLContent
from core.base_agent import BaseAgent
from tools.read_discord_messages import ReadDiscordMessages
from tools.write_discord_message import WriteDiscordMessage
import discord

class TerminalAgent(BaseAgent):
    """Terminal interface that understands natural language and executes tool commands."""
    
    command_history: List[str] = []
    max_results_per_query: int = 2
    tools: ClassVar[List[Type[BaseTool]]] = [
        DuckDuckGoSearch, 
        ParseURLContent, 
        ReadDiscordMessages, 
        WriteDiscordMessage
    ]
    
    # Initialize the Discord client
    def __init__(self, discord_client: discord.Client):
        super().__init__()
        self.discord_client = discord_client
        # Instantiate tools with the Discord client
        self.read_discord_messages = ReadDiscordMessages(self.discord_client)
        self.write_discord_message = WriteDiscordMessage(self.discord_client)

    system_prompt: str = """
You are the Terminal of Truth, an AI terminal interface that acts as a wise assistant to users to help them achieve their goals using natural language.

Your role is to:
1. Understand the user's intent from their natural language input.
2. Help them accomplish their goals using the available tools.
3. Maintain context of the conversation and previous commands.
4. Think step by step about complex requests.
5. Be proactive in suggesting related actions that might help.

When using tools:
1. Consider the context of previous commands.
2. Break down complex requests into steps.
3. Explain what you're doing and why.
4. Confirm success or explain failures.

For Discord operations:
- Use `read_discord_messages` to view recent messages. Parameters: `channel_id`, `limit` (default 10).
- Use `write_discord_message` to post new messages. Parameters: `channel_id`, `message`. Cooldown of 10 seconds applies.
- Always confirm message posting success.
- Show message content after posting.
"""

    @prompt_template("""
SYSTEM:
{self.system_prompt}
The current date is {current_date}.

Previous command context: {self.command_history}
Current user request: {query}

You have access to the following tools:
{self._get_tool_descriptions()}

USER: {query}
""")
    async def _stream(self, query: str) -> BaseDynamicConfig:
        """Override stream to include command history and tool descriptions."""
        return {
            "tools": self.tools,
            "computed_fields": {
                "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        tool_descriptions = []
        for tool in self.tools:
            params = ""
            if hasattr(tool, 'parameters'):
                params = f" (parameters: {', '.join(tool.parameters.keys())})"
            tool_descriptions.append(f"- `{tool.name}`: {tool.description}{params}")
        return "\n".join(tool_descriptions)

    async def _step(self, question: str):
        response = await self._stream(question)
        tools_and_outputs = []
        async for chunk, tool in response:
            if tool:
                print(f"\nExecuting: {tool.name} with parameters: {tool.args}")
                tool_instance = getattr(self, tool.name)
                output = await tool_instance(**tool.args)
                tools_and_outputs.append((tool, output))
            else:
                print(chunk.content, end="", flush=True)
        
        if response.user_message_param:
            self.messages.append(response.user_message_param)
        self.messages.append(response.message_param)
        
        if tools_and_outputs:
            self.messages += response.tool_message_params(tools_and_outputs)
            await self._step("")

    async def run(self):
        print("\nTerminal Agent initialized. Type 'exit' to quit.")
        print("Available tools:", ", ".join(tool.name for tool in self.tools))
        print("How can I help you today?\n")
        
        while True:
            question = input("\n(User): ")
            if question.lower() == "exit":
                print("\nTerminal Agent shutting down. Goodbye!")
                break
            print("(Terminal): ", end="", flush=True)
            await self._step(question)

# Entry point for running the Terminal Agent
if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.messages = True
    discord_client = discord.Client(intents=intents)

    agent = TerminalAgent(discord_client)

    async def start_bot():
        await discord_client.start('YOUR_DISCORD_BOT_TOKEN')

    loop = asyncio.get_event_loop()
    loop.create_task(start_bot())
    loop.run_until_complete(agent.run())