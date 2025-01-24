"""LLM foundation node type."""

import os
import sys
import asyncio
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, BaseModel

# Add parent directories to path if running directly
if __name__ == "__main__" and __package__ is None:
    file = os.path.abspath(__file__)
    parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
    sys.path.insert(0, parent)

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import Node, NodeState
from mirascope.core import BaseMessageParam, Messages, prompt_template

class MockAgent(BaseModel):
    """Mock agent for testing."""
    async def get_response(self, messages: List[BaseMessageParam]) -> str:
        """Return a mock response."""
        return f"Mock response to: {messages[-1].content}"

class LLMNode(Node):
    """Base class for nodes that use LLM capabilities.
    
    This node type adds LLM-specific functionality on top of the base Node class.
    It handles prompt formatting and LLM interaction through the BaseAgent.
    
    Attributes:
        runtime_config: Configuration for the LLM runtime
        prompt_template: Mirascope prompt template function for generating LLM prompts
        system_prompt: Optional system prompt to use for all messages
    """
    
    runtime_config: Optional[Any] = None
    prompt_template: Any = None  # Will hold the prompt template function
    system_prompt: Optional[str] = None
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process node using LLM.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
            
        The node will:
        1. Format prompt using Mirascope template and state results
        2. Send the messages to the LLM agent
        3. Store the response in results
        4. Return the next node ID based on default path
        """
        try:
            # Get prompt template result
            template_result = self.prompt_template(**state.results)
            
            # Handle dynamic config case
            if isinstance(template_result, dict):
                messages = template_result.get("messages", [])
                computed_fields = template_result.get("computed_fields", {})
                state.results.update(computed_fields)
            else:
                messages = template_result
            
            # Add system prompt if specified
            if self.system_prompt:
                messages = [Messages.System(self.system_prompt)] + messages
            
            # Create agent if needed
            if not hasattr(self, "_agent"):
                self._agent = BaseAgent(runtime_config=self.runtime_config)
            
            # Get LLM response
            response = await self._agent._step(messages)
            
            # Store result
            state.results[self.id] = {
                "response": response,
                "messages": messages
            }
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            state.results[self.id] = {"error": str(e)}
            return self.next_nodes.get("error")

async def test_llm_node():
    """Test LLM node functionality."""
    print("\nTesting LLMNode...")
    
    # Create test prompt template
    @prompt_template()
    def test_prompt(input_text: str) -> Messages.Type:
        return Messages.User(f"Summarize this in one sentence: {input_text}")
    
    # Create a test node
    node = LLMNode(
        id="test_llm",
        prompt_template=test_prompt,
        system_prompt="You are a helpful assistant that summarizes text concisely.",
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