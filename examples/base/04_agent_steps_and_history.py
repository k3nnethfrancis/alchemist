"""Example of agent steps and conversation history.

This example demonstrates:
1. Using _step() for a single interaction
2. Accessing and viewing conversation history
"""

import asyncio
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.logging import configure_logging, LogLevel
from alchemist.ai.prompts.persona import KEN_E

async def main():
    """Run the example."""
    # Configure logging
    configure_logging(default_level=LogLevel.INFO)
    
    # Create agent with Ken-E persona
    agent = BaseAgent(persona=KEN_E)
    
    # Ask a question and get response
    question = "What are three interesting facts about quantum computing?"
    print(f"\nAsking: {question}")
    
    response = await agent._step(question)
    print(f"\nResponse: {response}")
    
    # Show the conversation history
    print(agent.history)

if __name__ == "__main__":
    asyncio.run(main()) 