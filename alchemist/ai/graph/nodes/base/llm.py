"""LLM node for graph-based workflows."""

import asyncio
from typing import Dict, Any, Optional, Union, List
from pydantic import Field

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.logging import get_logger, LogComponent
from alchemist.ai.graph.nodes.base.node import Node, NodeState
from mirascope.core import BaseMessageParam, Messages, prompt_template

# Get logger for node operations
logger = get_logger(LogComponent.NODES)

class LLMNode(Node):
    """
    Node that processes input through an LLM agent.
    
    Attributes:
        prompt_template: Template function for generating prompts
        prompt: Optional raw prompt string (used if no template)
        system_prompt: Optional system prompt for the agent
        agent: Optional agent instance (created if not provided)
        runtime_config: Optional runtime configuration
    """
    
    prompt_template: Optional[Any] = None  # Function decorated with @prompt_template
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    agent: Optional[BaseAgent] = None
    runtime_config: Dict[str, Any] = Field(default_factory=dict)
    
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Process state through LLM and store response.
        
        Args:
            state: Current node state
            
        Returns:
            str: Next node ID
        """
        try:
            # Get or create agent
            agent = self.agent or BaseAgent(
                system_prompt=self.system_prompt,
                **self.runtime_config
            )
            
            # Get prompt using template or raw string
            if self.prompt_template:
                # Extract data for template
                template_data = {}
                template_data.update(state.data)
                for node_id, result in state.results.items():
                    if isinstance(result, dict):
                        template_data.update(result)
                    else:
                        template_data[f"{node_id}_result"] = result
                
                # Generate messages using template
                messages = self.prompt_template(**template_data)
                response = await agent.get_response(messages)
            else:
                # Use raw prompt string
                formatted_prompt = self._format_prompt(state)
                response = await agent.get_response(formatted_prompt)
            
            # Store result
            state.results[self.id] = {"response": response}
            
            return self.get_next_node()
            
        except Exception as e:
            logger.error(f"Error in LLM node {self.id}: {str(e)}")
            state.errors[self.id] = str(e)
            return self.get_next_node("error")
    
    def _format_prompt(self, state: NodeState) -> str:
        """Format prompt template with state data."""
        try:
            # Get data from state results or data
            format_data = {}
            format_data.update(state.data)
            
            # Add results from previous nodes
            for node_id, result in state.results.items():
                if isinstance(result, dict):
                    format_data.update(result)
                else:
                    format_data[f"{node_id}_result"] = result
            
            return self.prompt.format(**format_data)
            
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt: {str(e)}")
            
    def validate(self) -> bool:
        """Validate node configuration."""
        if not (self.prompt or self.prompt_template):
            return False
        return super().validate()

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