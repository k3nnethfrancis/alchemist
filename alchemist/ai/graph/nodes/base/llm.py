"""LLM foundation node type."""

import os
import sys
import asyncio
from typing import Any, Dict, Optional
from pydantic import Field, BaseModel

# Add parent directories to path if running directly
if __name__ == "__main__" and __package__ is None:
    file = os.path.abspath(__file__)
    parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
    sys.path.insert(0, parent)

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import Node, NodeState

class MockAgent(BaseModel):
    """Mock agent for testing."""
    async def get_response(self, prompt: str) -> str:
        """Return a mock response."""
        return f"Mock response to: {prompt}"

class LLMNode(Node):
    """Base class for nodes that use LLM capabilities.
    
    This node type adds LLM-specific functionality on top of the base Node class.
    It handles prompt formatting and LLM interaction through the BaseAgent.
    
    Attributes:
        agent: The LLM agent to use for processing
        prompt: Template string for generating LLM prompts
    """
    
    agent: BaseAgent = Field(default_factory=BaseAgent)
    prompt: str = ""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process node using LLM.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
            
        The node will:
        1. Format prompt template using state results
        2. Send the prompt to the LLM agent
        3. Store the response in results
        4. Return the next node ID based on default path
        """
        try:
            # Format prompt with state data
            formatted_prompt = self.prompt.format(**state.results)
            
            # Get LLM response using _step
            response = await self.agent._step(formatted_prompt)
            
            # Store result
            state.results[self.id] = {"response": response}
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            state.results[self.id] = {"error": str(e)}
            return self.next_nodes.get("error")

async def test_llm_node():
    """Test LLM node functionality."""
    print("\nTesting LLMNode...")
    
    # Create a test node
    node = LLMNode(
        id="test_llm",
        prompt="Summarize this in one sentence: {input_text}",
        next_nodes={"default": "next_node", "error": "error_node"}
    )
    
    # Create test state with input matching prompt template
    state = NodeState()
    state.results["input_text"] = "The quick brown fox jumps over the lazy dog. " \
                                 "This classic pangram contains every letter of the alphabet. " \
                                 "It has been used by typists and designers for years to showcase fonts."
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next_node", f"Expected 'next_node', got {next_id}"
    assert "response" in state.results["test_llm"], "No response in results"
    print(f"LLM Response: {state.results['test_llm']['response']}")
    print("LLMNode test passed!")

if __name__ == "__main__":
    print(f"Running test from: {__file__}")
    asyncio.run(test_llm_node()) 