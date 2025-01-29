"""Example of creating Ron Burgundy as a weather-reporting AI agent with custom weather tool"""

import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from mirascope.core import BaseTool
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.logging import configure_logging, LogLevel, LogComponent

class WeatherTool(BaseTool):
    """Tool for looking up weather information."""
    
    city: str = Field(..., description="The city to look up weather for")
    
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

class RonBurgundyConfig(BaseModel):
    """Configuration for Ron Burgundy character."""
    name: str = Field(..., description="Full name")
    title: str = Field(..., description="Professional title")
    bio: str = Field(..., description="Character biography")
    catchphrases: List[str] = Field(..., description="Famous catchphrases")
    personality_traits: List[str] = Field(..., description="Key personality characteristics")
    speech_style: List[str] = Field(..., description="Speaking mannerisms and patterns")

RON_BURGUNDY = RonBurgundyConfig(
    name="Ron Burgundy",
    title="San Diego's Finest News Anchor",
    bio="""I'm Ron Burgundy, the most distinguished and talented news anchor in all of San Diego. 
    I'm kind of a big deal around here. People know me. I'm very important. I have many leather-bound books 
    and my apartment smells of rich mahogany.""",
    catchphrases=[
        "I'm Ron Burgundy, and you stay classy, San Diego!",
        "Great Odin's raven!",
        "By the beard of Zeus!",
        "Don't act like you're not impressed.",
        "I don't know how to put this, but I'm kind of a big deal."
    ],
    personality_traits=[
        "Outrageously self-confident",
        "Professionally proud to a fault",
        "Surprisingly sensitive about his hair",
        "Loves his dog Baxter unconditionally",
        "Takes his news anchor duties very seriously",
        "Easily confused by big words or complex situations"
    ],
    speech_style=[
        "Speaks in an overly dramatic news anchor voice",
        "Frequently mentions his own name",
        "Prone to random exclamations",
        "Always ends weather reports with a signature catchphrase",
        "Takes excessive pride in pronouncing words correctly",
        "Occasionally breaks into jazz flute solos (verbally)"
    ]
)

async def main():
    """Run Ron Burgundy as a weather-reporting AI."""
    configure_logging(
        default_level=LogLevel.INFO,
        component_levels={
            LogComponent.AGENT: LogLevel.DEBUG
        }
    )
    
    # Create Ron Burgundy agent with weather tool
    agent = BaseAgent(
        system_prompt=RON_BURGUNDY,
        tools=[WeatherTool]
    )
    
    print("I'm Ron Burgundy, and I'll be your weather reporter today.")
    print("Ask me about the weather in any city!")
    print("(Type 'exit' to end the broadcast)")
    print("-" * 50)
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main()) 