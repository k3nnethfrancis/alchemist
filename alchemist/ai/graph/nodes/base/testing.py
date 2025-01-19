"""Testing utilities for graph nodes.

This module provides mock implementations and testing utilities for the graph system.
These are used in both unit tests and functional tests to verify behavior without
requiring external dependencies.
"""

from typing import Dict, Any
from pydantic import BaseModel

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.tools import BaseTool

class MockAgent(BaseAgent):
    """Mock agent for testing LLM nodes.
    
    This agent returns predictable responses based on the input prompt.
    It can be configured with specific responses for testing different scenarios.
    """
    
    def __init__(self, responses: Dict[str, str] = None):
        """Initialize mock agent.
        
        Args:
            responses: Optional mapping of prompts to responses
        """
        super().__init__()
        self.responses = responses or {}
        
    async def get_response(self, prompt: str) -> str:
        """Return a mock response.
        
        If the prompt matches a configured response, return that.
        Otherwise return a generic mock response.
        """
        if prompt in self.responses:
            return self.responses[prompt]
        return f"Mock response to: {prompt}"

class MockTool(BaseTool, BaseModel):
    """Mock tool for testing tool nodes.
    
    This tool returns predictable results based on the input arguments.
    It can be configured with specific results for testing different scenarios.
    """
    
    name: str = "mock_tool"
    args: Dict[str, Any] = {}
    results: Dict[str, str] = {}
    
    async def execute(self, **kwargs) -> str:
        """Return a mock result.
        
        If the args match a configured result, return that.
        Otherwise return a generic mock result.
        """
        args_key = str(sorted(kwargs.items()))
        if args_key in self.results:
            return self.results[args_key]
        return f"Mock result for {self.name} with args {kwargs}"

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def test_mock_agent():
        """Test mock agent functionality."""
        print("\nTesting MockAgent...")
        
        # Test default behavior
        agent = MockAgent()
        response = await agent.get_response("Hello")
        print(f"Default response: {response}")
        
        # Test configured responses
        agent = MockAgent({
            "What is 2+2?": "4",
            "Is this good?": "YES"
        })
        response = await agent.get_response("What is 2+2?")
        print(f"Configured response: {response}")
    
    async def test_mock_tool():
        """Test mock tool functionality."""
        print("\nTesting MockTool...")
        
        # Test default behavior
        tool = MockTool()
        result = await tool.execute(operation="test")
        print(f"Default result: {result}")
        
        # Test configured results
        tool = MockTool(
            name="calculator",
            results={
                str([("operation", "add"), ("numbers", [1,2,3])]): "6",
                str([("operation", "multiply"), ("numbers", [2,3])]): "6"
            }
        )
        result = await tool.execute(operation="add", numbers=[1,2,3])
        print(f"Configured result: {result}")
    
    async def run_tests():
        await test_mock_agent()
        await test_mock_tool()
    
    print("Running mock tests...")
    asyncio.run(run_tests()) 