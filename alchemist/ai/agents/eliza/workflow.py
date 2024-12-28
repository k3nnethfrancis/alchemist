"""Eliza workflow implementation."""

import logging
from typing import Optional

from alchemist.ai.graph.base import Graph
from alchemist.ai.graph.nodes.base import NodeState, NodeContext, LLMNode
from alchemist.ai.graph.nodes.decisions import MultiChoiceNode
from alchemist.ai.base.agent import BaseAgent

logger = logging.getLogger(__name__)

class ResponseNode(LLMNode):
    """Node that generates Eliza-style responses.
    
    This node uses the BaseAgent's LLM capabilities to generate
    therapeutic responses in the style of ELIZA.
    """
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Generate therapeutic response.
        
        Args:
            state: Current execution state with message context
            
        Returns:
            Optional[str]: Next node ID or None on error
        """
        try:
            # Get last message
            last_message = state.context.metadata.get("current_message", "")
            logger.debug(f"Generating response for message: {last_message}")
            
            # Generate response
            messages = [
                {"role": "system", "content": "You are ELIZA, a therapeutic chatbot."},
                {"role": "user", "content": last_message}
            ]
            
            response = await self._call_llm(messages)
            if not response:
                logger.error("Failed to generate response")
                return self.next_nodes.get("error")
                
            logger.debug(f"Generated response: {response}")
            
            # Store result
            state.results[self.id] = {
                "response": response,
                "message": last_message
            }
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return self.next_nodes.get("error")

def create_eliza_workflow() -> Graph:
    """Create the Eliza conversation workflow.
    
    This function creates a graph-based workflow that:
    1. Analyzes incoming messages
    2. Decides whether to respond
    3. Generates therapeutic responses when appropriate
    
    Returns:
        Graph: The configured workflow graph
    """
    graph = Graph()
    
    # Create base agent for LLM nodes with proper Mirascope setup
    agent = BaseAgent(provider="openai", model="gpt-4o-mini")
    
    # Create nodes
    analyze = MultiChoiceNode(
        id="analyze_message",
        agent=agent,
        prompt=(
            "Analyze this message and decide whether to respond:\n"
            "Message: {current_message}\n"
            "Is mention: {is_mention}\n"
            "Choose: respond or wait"
        ),
        choices=["respond", "wait"],
        next_nodes={
            "respond": "generate_response",
            "wait": "wait_node",
            "error": "wait_node"  # On error, just wait
        }
    )
    
    respond = ResponseNode(
        id="generate_response",
        agent=agent,
        next_nodes={
            "default": "wait_node",  # After responding, go to wait
            "error": "wait_node"  # On error, just wait
        }
    )
    
    wait = LLMNode(
        id="wait_node",
        agent=agent,
        next_nodes={}  # End node
    )
    
    # Add to graph
    graph.add_node(analyze)
    graph.add_node(respond)
    graph.add_node(wait)
    
    # Add entry point
    graph.add_entry_point("main", "analyze_message")
    
    # Log graph creation
    logger.info("Created Eliza workflow graph:")
    logger.info(f"- Analyze node: {analyze.id}")
    logger.info(f"- Generate node: {respond.id}")
    logger.info(f"- Wait node: {wait.id}")
    
    return graph 