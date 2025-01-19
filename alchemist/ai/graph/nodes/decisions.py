"""Decision node implementations for LLM-based choices."""

import logging
from typing import Dict, Any, Optional, List
from pydantic import Field

from alchemist.ai.graph.base import NodeState
from alchemist.ai.graph.nodes.base.llm import LLMNode
from alchemist.ai.base.agent import BaseAgent

logger = logging.getLogger(__name__)

class BinaryDecisionNode(LLMNode):
    """Node that makes a binary decision using LLM.
    
    This node uses an LLM to make yes/no decisions based on a prompt template.
    The prompt is formatted with context metadata before being sent to the LLM.
    
    Attributes:
        prompt: Template string for generating the decision prompt
        agent: The LLM agent to use (inherited from LLMNode)
    """
    
    prompt: str = Field(default="")
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process state and return yes/no path.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute based on yes/no decision
            
        The node will:
        1. Format the prompt with context metadata
        2. Get a yes/no decision from the LLM
        3. Store the decision and response
        4. Return the next node ID based on the decision
        """
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
                "response": response,
                "timestamp": await state.context.get_context("time")
            }
            
            # Return next node based on decision
            next_node = self.next_nodes.get(decision)
            logger.debug(f"Selected next node: {next_node}")
            return next_node
            
        except Exception as e:
            logger.error(f"Error in binary decision: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "prompt": formatted_prompt if 'formatted_prompt' in locals() else None
            }
            return self.next_nodes.get("error")

class MultiChoiceNode(LLMNode):
    """Node that makes a multi-choice decision using LLM.
    
    This node uses an LLM to select from multiple choices based on a prompt template.
    The prompt is formatted with context metadata before being sent to the LLM.
    
    Attributes:
        choices: List of valid choices
        prompt: Template string for generating the decision prompt
        agent: The LLM agent to use (inherited from LLMNode)
    """
    
    choices: List[str] = Field(default_factory=list)
    prompt: str = Field(default="")
    
    def __init__(self, **data):
        """Initialize with required agent if not provided."""
        if "agent" not in data:
            data["agent"] = BaseAgent()
        super().__init__(**data)
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process state and return selected choice path.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute based on selected choice
            
        The node will:
        1. Format the prompt with context metadata
        2. Get a choice selection from the LLM
        3. Store the choice and response
        4. Return the next node ID based on the choice
        """
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
            
            response = await self.agent.get_response(messages)
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
                "response": response,
                "choices": self.choices,
                "timestamp": await state.context.get_context("time")
            }
            
            # Return next node based on choice
            next_node = self.next_nodes.get(choice)
            logger.debug(f"Selected next node: {next_node}")
            return next_node
            
        except Exception as e:
            logger.error(f"Error in multi-choice decision: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "prompt": formatted_prompt if 'formatted_prompt' in locals() else None,
                "choices": self.choices
            }
            return self.next_nodes.get("error") 