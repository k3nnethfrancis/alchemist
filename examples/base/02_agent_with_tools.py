"""Example of creating an agent with tool integration.

This example demonstrates using BaseAgent for tool-driven tasks:
1. Using built-in tools (Calculator)
2. Creating custom tools (Weather)
3. Combining multiple tools in one agent
4. Handling tool execution and results

Best for:
- Tool integration
- Custom tool creation
- Multi-tool agents
- Task-specific automation
"""

import asyncio
from typing import Dict, Any
from pydantic import BaseModel, Field

from mirascope.core import BaseTool
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.base.logging import configure_logging, LogLevel, LogComponent
from alchemist.ai.tools.calculator import CalculatorTool

# Define a custom tool for weather lookup
class WeatherTool(BaseTool):
    """Tool for looking up weather information."""
    
    city: str = Field(
        ...,
        description="The city to look up weather for"
    )
    
    @classmethod
    def _name(cls) -> str:
        """Get the tool's name for LLM function calling."""
        return "weather"
        
    @classmethod
    def _description(cls) -> str:
        """Get the tool's description for LLM function calling."""
        return "Look up current weather for a city"
        
    def call(self) -> Dict[str, Any]:
        """Mock weather lookup."""
        # In a real implementation, this would call a weather API
        return {
            "city": self.city,
            "temperature": "72°F",
            "condition": "Sunny"
        }

# Define a tool-using assistant persona
TOOL_ASSISTANT = {
    "id": "tool-assistant-v1",
    "name": "ToolGPT",
    "nickname": "Tool",
    "bio": """I am an assistant that demonstrates tool usage.
I can help with calculations and weather lookups.""",
    "personality": {
        "traits": {
            "neuroticism": 0.2,      # Stable and reliable
            "extraversion": 0.6,      # Engaging but focused
            "openness": 0.8,         # Adaptable to tasks
            "agreeableness": 0.7,    # Helpful while focused
            "conscientiousness": 0.9  # Detail-oriented
        },
        "stats": {
            "intelligence": 0.9,      # Strong problem-solving
            "wisdom": 0.8,           # Good judgment
            "charisma": 0.7,         # Clear communication
            "authenticity": 1.0,     # Transparent operation
            "adaptability": 0.8,     # Flexible to needs
            "reliability": 0.9       # Consistent execution
        }
    },
    "lore": [
        "Expert in using tools",
        "Combines multiple tools effectively",
        "Provides clear results",
        "Explains tool usage",
        "Helps with practical tasks"
    ],
    "style": {
        "all": [
            "Uses clear language",
            "Explains tool selection",
            "Shows tool results",
            "Provides helpful context",
            "Maintains task focus"
        ],
        "chat": [
            "Confirms task understanding",
            "Updates on tool usage",
            "Reports results clearly",
            "Suggests relevant tools",
            "Maintains helpful tone"
        ]
    }
}

async def main():
    """Run a tool-focused session."""
    # Configure logging for tool usage tracking
    configure_logging(
        default_level=LogLevel.INFO,
        component_levels={
            LogComponent.AGENT: LogLevel.DEBUG  # Detailed logs for tool usage
        }
    )
    
    # Create agent with both built-in and custom tools
    agent = BaseAgent(
        persona=PersonaConfig(**TOOL_ASSISTANT),
        tools=[CalculatorTool, WeatherTool]  # Combine built-in and custom tools
    )
    
    print("\nInitialized Tool Assistant")
    print("I can help you with calculations and weather lookups.")
    print("\nAvailable tools:")
    print("1. Calculator - Perform mathematical calculations")
    print("2. Weather - Look up weather for a city")
    print("\nTry asking:")
    print("- What is 123 * 456?")
    print("- What's the weather in San Francisco?")
    print("- Can you calculate 15% of 85?")
    print("\nType 'exit' or 'quit' to end")
    print("-" * 50)
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        # Execute with tool support
        result = await agent._step(query)
        print(f"\nAssistant: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 