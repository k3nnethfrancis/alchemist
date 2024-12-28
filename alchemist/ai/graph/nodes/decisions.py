"""Decision node implementations."""

import logging
from typing import Dict, Any, Optional, List
from pydantic import Field

from alchemist.ai.graph.nodes.base import Node, NodeState, LLMNode
from alchemist.ai.base.agent import BaseAgent

logger = logging.getLogger(__name__)

class BinaryDecisionNode(LLMNode):
    """Node that makes a binary decision."""
    
    prompt: str = Field(default="")
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process state and return yes/no path."""
        try:
            # Format prompt with state context
            try:
                formatted_prompt = self.prompt.format(**state.context.metadata)
                logger.debug(f"Formatted prompt: {formatted_prompt}")
            except KeyError as e:
                logger.error(f"Missing key in context metadata: {e}")
                logger.debug(f"Available context: {state.context.metadata}")
                return self.next_nodes.get("error")
            
            # Get LLM decision using base class method
            system_prompt = "You are making a yes/no decision. Respond with only 'yes' or 'no'."
            full_prompt = f"{system_prompt}\n\n{formatted_prompt}"
            
            response = await self.agent.get_response(full_prompt)
            if not response:
                logger.error("No response from LLM")
                return self.next_nodes.get("error")
                
            # Parse response to get decision
            decision = response.strip().lower()
            logger.debug(f"Raw decision from LLM: {decision}")
            
            if decision not in ["yes", "no"]:
                logger.warning(f"Invalid decision '{decision}', defaulting to no")
                decision = "no"
                
            # Store result
            state.results[self.id] = {
                "decision": decision,
                "prompt": formatted_prompt,
                "response": response
            }
            
            # Return next node based on decision
            next_node = self.next_nodes.get(decision)
            logger.debug(f"Selected next node: {next_node}")
            return next_node
            
        except Exception as e:
            logger.error(f"Error in binary decision: {str(e)}")
            return self.next_nodes.get("error")

class MultiChoiceNode(LLMNode):
    """Node that makes a multi-choice decision using LLM."""
    
    choices: List[str] = Field(default_factory=list)
    prompt: str = Field(default="")
    
    def __init__(self, **data):
        """Initialize with required agent if not provided."""
        if "agent" not in data:
            data["agent"] = BaseAgent()
        super().__init__(**data)
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process state and return selected choice path."""
        try:
            # Format prompt with state context
            try:
                formatted_prompt = self.prompt.format(**state.context.metadata)
                logger.debug(f"Formatted prompt: {formatted_prompt}")
            except KeyError as e:
                logger.error(f"Missing key in context metadata: {e}")
                logger.debug(f"Available context: {state.context.metadata}")
                return self.next_nodes.get("error")
            
            # Get LLM decision
            messages = [
                {"role": "system", "content": f"You are making a decision between multiple choices: {', '.join(self.choices)}. Respond with only one of these exact choices."},
                {"role": "user", "content": formatted_prompt}
            ]
            
            response = await self._call_llm(messages)
            if not response:
                logger.error("No response from LLM")
                return self.next_nodes.get("error")
                
            # Parse response to get choice
            choice = response.strip()
            logger.debug(f"Raw choice from LLM: {choice}")
            
            if choice not in self.choices:
                logger.warning(f"Invalid choice '{choice}', defaulting to first choice")
                choice = self.choices[0]
                
            # Store result
            state.results[self.id] = {
                "choice": choice,
                "prompt": formatted_prompt,
                "response": response
            }
            
            # Return next node based on choice
            next_node = self.next_nodes.get(choice)
            logger.debug(f"Selected next node: {next_node}")
            return next_node
            
        except Exception as e:
            logger.error(f"Error in multi-choice decision: {str(e)}")
            return self.next_nodes.get("error") 