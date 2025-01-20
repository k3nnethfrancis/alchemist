"""Decision node implementations for LLM-based choices."""

import logging
from typing import Dict, Any, Optional, List
from pydantic import Field
import asyncio

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

async def test_binary_decision_node():
    """Test binary decision node functionality."""
    print("\nTesting BinaryDecisionNode...")
    
    # Create test agent that always says yes
    class TestAgent:
        async def get_response(self, prompt: str) -> str:
            return "yes"
    
    # Create test node
    node = BinaryDecisionNode(
        id="test_binary",
        prompt="Is {value} greater than 10?",
        agent=TestAgent(),
        next_nodes={
            "yes": "yes_node",
            "no": "no_node",
            "error": "error_node"
        }
    )
    
    # Create test state
    state = NodeState()
    state.context.metadata["value"] = 15
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "yes_node", f"Expected 'yes_node', got {next_id}"
    assert state.results["test_binary"]["decision"] == "yes", "Incorrect decision"
    print("BinaryDecisionNode test passed!")

async def test_multi_choice_node():
    """Test multi-choice node functionality."""
    print("\nTesting MultiChoiceNode...")
    
    # Create test agent that always picks first choice
    class TestAgent:
        async def get_response(self, messages: list) -> str:
            return "tech"
    
    # Create test node
    node = MultiChoiceNode(
        id="test_multi",
        prompt="Categorize this content about {topic}",
        choices=["tech", "news", "other"],
        agent=TestAgent(),
        next_nodes={
            "tech": "tech_node",
            "news": "news_node",
            "other": "other_node",
            "error": "error_node"
        }
    )
    
    # Create test state
    state = NodeState()
    state.context.metadata["topic"] = "artificial intelligence"
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "tech_node", f"Expected 'tech_node', got {next_id}"
    assert state.results["test_multi"]["choice"] == "tech", "Incorrect choice"
    print("MultiChoiceNode test passed!")

async def test_error_handling():
    """Test error handling in decision nodes."""
    print("\nTesting error handling...")
    
    # Test missing context metadata
    node = BinaryDecisionNode(
        id="test_error",
        prompt="This will fail: {missing_key}",
        next_nodes={"error": "error_node"}
    )
    
    state = NodeState()
    next_id = await node.process(state)
    
    assert next_id == "error_node", "Error not handled correctly"
    assert "error" in state.results["test_error"], "Error not recorded in state"
    print("Error handling test passed!")

async def run_all_tests():
    """Run all decision node tests."""
    print("Running decision node tests...")
    await test_binary_decision_node()
    await test_multi_choice_node()
    await test_error_handling()
    print("\nAll decision node tests passed!")

if __name__ == "__main__":
    print(f"Running tests from: {__file__}")
    asyncio.run(run_all_tests()) 