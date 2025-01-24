"""Example of extending agent capabilities using the graph system.

This example shows how to use the graph system to add reflection and context
enhancement to a base agent's chat functionality.
"""

import asyncio
import logging
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

from mirascope.core import Messages, prompt_template

from alchemist.ai.base.runtime import RuntimeConfig
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import Graph, NodeContext, NodeState
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.prompts.persona import ALCHEMIST

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

@prompt_template()
def reflection_prompt(messages: List[Messages.Type], **kwargs) -> Messages.Type:
    """Create a reflection prompt to analyze conversation and enhance response.
    
    Args:
        messages: List of conversation messages
        **kwargs: Additional arguments from state
        
    Returns:
        Messages object containing the reflection prompt
    """
    # Create reflection prompt
    reflection_text = """
    Analyze the conversation so far and consider:
    1. Key themes and topics discussed
    2. User's implicit needs or concerns
    3. Potential areas to provide additional value
    4. Relevant context to incorporate
    
    Provide a concise reflection to enhance the response.
    """
    
    # Add conversation history
    return [
        Messages.System(reflection_text),
        *messages
    ]

class ReflectionNode(LLMNode):
    """Node that adds reflection capabilities to agent responses."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process messages through reflection before agent response.
        
        Args:
            state: Current node state with messages
            
        Returns:
            ID of next node (agent)
        """
        try:
            # Get messages from state
            messages = state.results.get("messages", [])
            if not messages:
                return self.next_nodes.get("default")
                
            # Generate reflection using LLM
            reflection = await self._agent._step(
                reflection_prompt(messages=messages)
            )
            
            # Add reflection to state
            state.results["reflection"] = reflection
            logger.info(f"\nReflection: {reflection}\n")
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            logger.error(f"Error in reflection: {str(e)}")
            state.results["reflection_error"] = str(e)
            return self.next_nodes.get("error")

class AgentNode(LLMNode):
    """Node that wraps a base agent with graph capabilities."""
    
    agent: BaseAgent
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process messages using the base agent.
        
        Args:
            state: Current node state with messages
            
        Returns:
            ID of next node or None if finished
        """
        try:
            # Get messages and reflection
            messages = state.results.get("messages", [])
            reflection = state.results.get("reflection")
            
            if reflection:
                # Add reflection as system message
                messages = [Messages.System(reflection)] + messages
            
            # Get agent response
            response = await self.agent._step(messages)
            
            # Store response
            state.results["response"] = response
            logger.info(f"Agent: {response}")
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            logger.error(f"Error in agent: {str(e)}")
            state.results["agent_error"] = str(e)
            return self.next_nodes.get("error")

async def main():
    """Run an enhanced agent chat using the graph system."""
    # Load environment variables
    load_dotenv()
    
    # Create base agent
    config = RuntimeConfig(
        provider="openpipe",
        model="gpt-4o-mini",
        persona=ALCHEMIST,
        tools=[]
    )
    agent = BaseAgent(runtime_config=config.model_dump())
    
    # Create nodes
    reflection = ReflectionNode(
        id="reflect",
        runtime_config=config.model_dump(),
        next_nodes={"default": "agent", "error": "agent"}
    )
    
    agent_node = AgentNode(
        id="agent",
        agent=agent,
        next_nodes={"default": None, "error": None}
    )
    
    # Create and configure graph
    graph = Graph()
    graph.add_node(reflection)
    graph.add_node(agent_node)
    graph.add_entry_point("start", "reflect")
    
    # Create state
    state = NodeState()
    
    # Run chat loop
    history = []
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            # Add message to history
            history.append(Messages.User(user_input))
            
            # Update state
            state.results["messages"] = history
            
            # Process through graph
            state = await graph.run("start", state)
            
            # Get response and update history
            response = state.results.get("response")
            if response:
                history.append(Messages.Assistant(response))
            
        except KeyboardInterrupt:
            print("\nChat session terminated by user.")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise 