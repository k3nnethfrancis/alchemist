"""Agent node for graph-based workflows.

This module implements a node that wraps BaseAgent functionality for use in graphs.
It supports:
    - Direct agent execution via _step()
    - Prompt templating and formatting
    - System prompts and runtime configuration
    - State management for conversation history

Usage:
------
>>> node = AgentNode(
...     id="reasoning_step",
...     prompt_template=my_prompt_func,
...     system_prompt="You are a helpful assistant...",
...     next_nodes={"default": "next_node", "error": "error_node"}
... )
>>> state = NodeState()
>>> next_id = await node.process(state)
"""

import asyncio
from typing import Dict, Any, Optional, Union, List
from pydantic import Field, model_validator
from datetime import datetime

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.logging import get_logger, LogComponent
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.state import NodeState, NodeStatus
from mirascope.core import BaseMessageParam, Messages, prompt_template
from alchemist.ai.base.logging import Colors

# Get logger for node operations
logger = get_logger(LogComponent.NODES)

class AgentNode(Node):
    """
    Node that processes input through an agent.
    
    This node wraps BaseAgent functionality for use in graph workflows. It can:
    - Use an existing agent or create one with provided configuration
    - Format prompts using templates or raw strings
    - Maintain conversation history in NodeState
    - Execute single-step or multi-step agent reasoning
    
    Attributes:
        prompt_template: Template function for generating prompts
        prompt: Optional raw prompt string (used if no template)
        system_prompt: Optional system prompt for the agent
        agent: Optional agent instance (created if not provided)
        runtime_config: Optional runtime configuration
    """
    
    prompt_template: Optional[Any] = Field(default=None)
    prompt: Optional[str] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None)
    agent: Optional[BaseAgent] = Field(default=None)
    runtime_config: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_agent_config(self) -> 'AgentNode':
        """Validate agent configuration."""
        if not (self.prompt or self.prompt_template):
            raise ValueError(f"AgentNode {self.id} requires either prompt or prompt_template")
        return self
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process state through agent and store response."""
        try:
            start_time = datetime.now()
            state.mark_status(self.id, NodeStatus.RUNNING)
            
            # Format the prompt using state data
            final_prompt = self._format_prompt(state)
            logger.debug(f"\n{Colors.BOLD}🤖 Node {self.id} Prompt:{Colors.RESET}\n{final_prompt}")
            
            # Use existing agent or create new one
            used_agent = self.agent or BaseAgent(
                system_prompt=self.system_prompt,
                **self.runtime_config
            )
            
            # Execute agent step
            response = await used_agent._step(final_prompt)
            logger.debug(f"\n{Colors.BOLD}📝 Node {self.id} Response:{Colors.RESET}\n{response}")
            
            # Store results
            state.results[self.id] = {
                "response": response,
                "timing": (datetime.now() - start_time).total_seconds()
            }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"\n{Colors.SUCCESS}✓ Node '{self.id}' completed in {elapsed:.2f}s{Colors.RESET}"
                f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}"
                f"\n{Colors.INFO}{response}{Colors.RESET}"
                f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}\n"
            )
            
            state.mark_status(self.id, NodeStatus.COMPLETED)
            return self.get_next_node()
            
        except Exception as e:
            logger.error(f"Error in AgentNode '{self.id}': {str(e)}")
            state.errors[self.id] = str(e)
            state.mark_status(self.id, NodeStatus.ERROR)
            return self.get_next_node("error")
    
    def _format_prompt(self, state: NodeState) -> str:
        """Format prompt using state data and input_map."""
        try:
            # Get mapped input data (from both data and results)
            input_data = self._prepare_input_data(state)
            
            if self.prompt_template:
                return self.prompt_template(**input_data)
            else:
                return self.prompt.format(**input_data)
                
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt: {str(e)}")

async def test_agent_node():
    """Test agent node functionality."""
    print("\nTesting AgentNode...")
    
    @prompt_template()
    def test_prompt(input_text: str) -> Messages.Type:
        return Messages.User(f"Summarize this in one sentence: {input_text}")
    
    node = AgentNode(
        id="test_agent",
        prompt_template=test_prompt,
        system_prompt="You are a helpful assistant that summarizes text concisely.",
        next_nodes={"default": "next_node", "error": "error_node"}
    )
    
    state = NodeState()
    state.results["input_text"] = (
        "The quick brown fox jumps over the lazy dog. "
        "This classic pangram contains every letter of the alphabet. "
        "It has been used by typists and designers for years to showcase fonts."
    )
    
    next_id = await node.process(state)
    
    assert next_id == "next_node", f"Expected 'next_node', got {next_id}"
    assert "response" in state.results["test_agent"], "No response in results"
    print(f"Agent Response: {state.results['test_agent']['response']}")
    print("AgentNode test passed!")

if __name__ == "__main__":
    print(f"Running test from: {__file__}")
    asyncio.run(test_agent_node()) 