"""Example of a simple Graph workflow using graph nodes.

This example demonstrates:
1. Waiting for messages
2. Reading Discord messages
3. Deciding whether to respond
4. Generating appropriate responses
"""

from typing import List, Dict, Any
from alchemist.ai.graph.base import Graph, NodeState
from alchemist.ai.graph.nodes.decisions import BinaryDecisionNode
from alchemist.ai.graph.nodes.actions import WaitNode, ToolNode
from alchemist.ai.graph.nodes.base import LLMNode

async def create_eliza_workflow() -> Graph:
    """Create a simple Eliza workflow."""
    
    # Create nodes
    wait_node = WaitNode(
        id="wait",
        seconds=60,
        next_nodes={"default": "fetch_messages"}
    )
    
    fetch_messages = ToolNode(
        id="fetch_messages",
        tool_name="discord.fetch_messages",
        tool_args={"limit": 10},
        next_nodes={"default": "should_respond"}
    )
    
    should_respond = BinaryDecisionNode(
        id="should_respond",
        prompt="Based on the recent messages, should we respond?",
        examples=[
            {
                "context": "User asked a direct question",
                "choice": "yes"
            },
            {
                "context": "Conversation is inactive",
                "choice": "no"
            }
        ],
        next_nodes={
            "yes": "generate_response",
            "no": "wait"
        }
    )
    
    generate_response = LLMNode(
        id="generate_response",
        prompt="Generate a helpful response to the user's message.",
        examples=[
            {
                "context": "User: I'm feeling sad",
                "response": "I hear that you're feeling sad. Can you tell me more about what's troubling you?"
            }
        ],
        next_nodes={"default": "wait"}
    )
    
    # Create graph
    graph = Graph()
    
    # Add nodes
    graph.add_node(wait_node)
    graph.add_node(fetch_messages)
    graph.add_node(should_respond)
    graph.add_node(generate_response)
    
    # Validate graph
    graph.validate()
    
    return graph

async def run_workflow():
    """Run the Eliza workflow."""
    # Create graph
    graph = await create_eliza_workflow()
    
    # Create initial state
    state = NodeState(
        context={
            "history": [],
            "metadata": {}
        },
        results={}
    )
    
    # Run graph starting at wait node
    await graph.run("wait", state)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_workflow()) 