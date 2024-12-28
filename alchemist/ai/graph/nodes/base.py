"""Base node implementations for the graph system."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import NodeState

class Node(BaseModel):
    """Base class for all graph nodes."""
    
    id: str
    next_nodes: Dict[str, Optional[str]] = Field(default_factory=dict)
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process node and return next node ID."""
        raise NotImplementedError

class LLMNode(Node):
    """Base class for nodes that use LLM capabilities."""
    
    agent: BaseAgent = Field(default_factory=lambda: BaseAgent(provider="openai"))
    prompt: str = ""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process node using LLM."""
        try:
            # Format prompt with state data
            formatted_prompt = self.prompt.format(**state.results)
            
            # Get LLM response
            response = await self.agent.get_response(formatted_prompt)
            
            # Store result
            state.results[self.id] = {"response": response}
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            state.results[self.id] = {"error": str(e)}
            return None 