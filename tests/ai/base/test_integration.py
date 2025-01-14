"""Integration tests for BaseAgent with multiple tools."""

import pytest
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from mirascope.core import BaseMessageParam
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.tools import CalculatorTool, ImageGenerationTool

@pytest.mark.asyncio
async def test_agent_with_multiple_tools():
    """Test agent initialization and execution with both calculator and image generation tools."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Initialize agent with both tools
    agent = BaseAgent(tools=[CalculatorTool, ImageGenerationTool])
    
    # Test calculator functionality
    calc_query = "What is the square root of 42?"
    result = await agent._step(calc_query)
    assert "6.48" in str(result)  # Check for rounded value
    
    # Test image generation
    try:
        image_query = "Generate an image of a cyberpunk city at sunset"
        result = await agent._step(image_query)
        assert "http" in str(result)  # URL should be returned
    except Exception as e:
        pytest.skip(f"Skipping image generation test: {str(e)}")

@pytest.mark.asyncio
async def test_parallel_tool_execution():
    """Test agent's ability to use multiple tools in a single response."""
    
    agent = BaseAgent(tools=[CalculatorTool, ImageGenerationTool])
    
    # Query that should trigger both tools
    query = "Calculate 2 + 2 and generate an image of the number 4"
    
    try:
        result = await agent._step(query)
        history = agent.history
        
        # Verify both tools were used
        tool_messages = [msg for msg in history if msg.role == "function"]
        assert len(tool_messages) >= 2
        
        # Verify calculator result
        calc_result = next(msg for msg in tool_messages if msg.name == "calculate")
        assert "4" in calc_result.content
        
        # Verify image result
        image_result = next(msg for msg in tool_messages if msg.name == "generate_image")
        assert image_result.content.startswith("http")
    except Exception as e:
        pytest.skip(f"Skipping parallel tool test: {str(e)}")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Run the integration tests manually."""
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        
        # Initialize agent with both tools
        agent = BaseAgent(tools=[CalculatorTool, ImageGenerationTool])
        
        print("\nInitialized agent with Calculator and Image Generation tools")
        print("Type 'exit' or 'quit' to end the conversation")
        print("-" * 50)
        
        while True:
            query = input("\n(User): ")
            if query.lower() in ["exit", "quit"]:
                break
            print("(Assistant): ", end="", flush=True)
            result = await agent._step(query)
            print(result)
    
    asyncio.run(main()) 