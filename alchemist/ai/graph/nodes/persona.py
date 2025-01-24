"""Persona-based LLM node implementation."""

from typing import Optional, Any

from alchemist.ai.graph.nodes.base.llm import LLMNode
from alchemist.ai.graph.base import NodeState
from alchemist.ai.prompts.base import PersonaConfig, create_system_prompt

class PersonaLLMNode(LLMNode):
    """LLM node that uses a persona configuration for system prompts.
    
    This node extends the base LLMNode to support persona-based interactions.
    It automatically configures the system prompt based on the provided persona.
    
    Attributes:
        persona: PersonaConfig object defining the AI persona
        runtime_config: Configuration for the LLM runtime
        prompt_template: Mirascope prompt template function for generating LLM prompts
    """
    
    persona: PersonaConfig
    
    def __init__(self, **data):
        """Initialize the node with a persona."""
        super().__init__(**data)
        # Set system prompt from persona
        self.system_prompt = create_system_prompt(self.persona)
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process node using persona-based LLM.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
        """
        # Add persona to state results for prompt templates
        state.results["persona"] = self.persona
        
        # Process using parent implementation
        return await super().process(state) 