"""Example of extending agent capabilities using the graph system.

This example shows how to use the graph system to add reflection and context
enhancement to a base agent's chat functionality.
"""

import asyncio
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from mirascope.core import Messages, prompt_template

from alchemist.ai.base.runtime import RuntimeConfig
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import Graph, NodeContext, NodeState
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.prompts.persona import ALCHEMIST
from alchemist.ai.base.logging import (
    configure_logging,
    LogComponent,
    LogLevel,
    LogFormat,
    get_logger,
    log_state
)

# Set up logging
log_dir = Path("logs/persona_chat")
log_dir.mkdir(parents=True, exist_ok=True)

configure_logging(
    default_level=LogLevel.INFO,
    component_levels={
        LogComponent.AGENT: LogLevel.DEBUG,
        LogComponent.GRAPH: LogLevel.DEBUG,
        LogComponent.NODES: LogLevel.DEBUG,
        LogComponent.WORKFLOW: LogLevel.DEBUG
    },
    format_string=LogFormat.DEBUG,
    log_file=str(log_dir / f"persona_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    enable_json=True
)

# Get component loggers
workflow_logger = get_logger(LogComponent.WORKFLOW)
graph_logger = get_logger(LogComponent.GRAPH)
node_logger = get_logger(LogComponent.NODES)

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
                node_logger.warning("No messages found in state")
                return self.next_nodes.get("default")
                
            # Log state before reflection
            node_logger.debug("State before reflection:")
            log_state(node_logger, state.dict())
                
            # Generate reflection using LLM
            reflection = await self._agent._step(
                reflection_prompt(messages=messages)
            )
            
            # Add reflection to state
            state.results["reflection"] = reflection
            node_logger.info(f"Generated reflection for conversation: {self.id}")
            node_logger.debug(f"Reflection content: {reflection}")
            
            # Log updated state
            node_logger.debug("State after reflection:")
            log_state(node_logger, state.dict())
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            node_logger.error(f"Error in reflection node {self.id}: {str(e)}", exc_info=True)
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
            
            # Log input state
            node_logger.debug("Agent node input state:")
            log_state(node_logger, {"messages": messages, "reflection": reflection})
            
            if reflection:
                # Add reflection as system message
                messages = [Messages.System(reflection)] + messages
                node_logger.debug("Added reflection to messages")
            
            # Get agent response
            response = await self.agent._step(messages)
            
            # Store response
            state.results["response"] = response
            node_logger.info(f"Generated agent response: {self.id}")
            node_logger.debug(f"Response content: {response}")
            
            # Log final state
            node_logger.debug("Agent node final state:")
            log_state(node_logger, state.dict())
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            node_logger.error(f"Error in agent node {self.id}: {str(e)}", exc_info=True)
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
    workflow_logger.info("Initialized agent with configuration")
    workflow_logger.debug(f"Agent config: {config.model_dump()}")
    
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
    workflow_logger.info("Graph configured with reflection and agent nodes")
    
    # Create state
    state = NodeState()
    
    # Run chat loop
    history = []
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                workflow_logger.info("Chat session ended by user")
                break
                
            # Add message to history
            history.append(Messages.User(user_input))
            workflow_logger.debug(f"Added user message to history: {user_input}")
            
            # Update state
            state.results["messages"] = history
            
            # Process through graph
            workflow_logger.info("Processing user input through graph")
            state = await graph.run("start", state)
            
            # Get response and update history
            response = state.results.get("response")
            if response:
                history.append(Messages.Assistant(response))
                print(f"\nAgent: {response}")
                workflow_logger.debug(f"Added agent response to history: {response}")
            
        except KeyboardInterrupt:
            workflow_logger.info("Chat session terminated by user interrupt")
            break
        except Exception as e:
            workflow_logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        workflow_logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise 