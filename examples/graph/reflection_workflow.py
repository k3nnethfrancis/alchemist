"""Example of a reflective chatbot using graph nodes with AUG_E persona."""

from typing import Dict, Any, Optional
from alchemist.ai.graph.base import Graph, NodeState, NodeContext
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.persona import AUG_E
from alchemist.ai.base.logging import configure_logging, LogComponent, LogLevel
import logging
import time
from datetime import datetime

# Configure logging - debug for graph, info for others
configure_logging(
    default_level=LogLevel.INFO,
    component_levels={
        LogComponent.GRAPH: LogLevel.DEBUG,
        LogComponent.NODES: LogLevel.DEBUG
    }
)

# Get module logger
logger = logging.getLogger(__name__)

class ThinkingNode(LLMNode):
    """Node that performs step-by-step thinking."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process thinking steps."""
        try:
            start_time = datetime.now()
            print("\nThinking...\n", flush=True)
            
            # Initialize step tracking if not present
            if "step_number" not in state.data:
                state.data["step_number"] = 1
                state.data["previous_steps"] = ""
            
            # Get response from LLM
            response = await self.agent.get_response(self.format_prompt(state))
            
            # Store response in results
            state.results[self.id] = {"response": response}
            
            # Update step tracking
            state.data["previous_steps"] += f"\nStep {state.data['step_number']}:\n{response}\n"
            state.data["step_number"] += 1
            
            # Log timing
            end_time = datetime.now()
            thinking_time = (end_time - start_time).total_seconds()
            print(f"[Step {state.data['step_number']-1} thinking time: {thinking_time:.2f} seconds]\n", flush=True)
            
            # Decide next node
            if "NEXT: final_answer" in response or state.data["step_number"] > 5:
                return "respond"
            return "think"
            
        except Exception as e:
            logger.error(f"Error in thinking node: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            return "respond"  # Fallback to respond on error

    def format_prompt(self, state: NodeState) -> str:
        """Format the prompt with state data."""
        return f"""TITLE: Step-by-step Analysis

        User Message: {state.results.get('input', {}).get('message', '')}
        Previous Steps: {state.data.get('previous_steps', '')}
        Current Step: {state.data.get('step_number', 1)}
        
        Provide your response in this format:
        TITLE: [title of the step]
        CONTENT: [your detailed reasoning]
        NEXT: [either "continue" or "final_answer"]
        
        Guidelines:
        - Use AT MOST 5 steps
        - Be aware of your limitations
        - Consider alternative perspectives
        - Test assumptions thoroughly
        - Be willing to be wrong and re-examine"""

async def create_reflection_workflow() -> Graph:
    """Create a reflective chatbot workflow."""
    logger.info("Creating reflection workflow...")
    
    # Create agent with AUG_E persona
    agent = BaseAgent(provider="openpipe", persona=AUG_E)
    
    think = ThinkingNode(
        id="think",
        agent=agent,
        next_nodes={"think": "think", "respond": "respond"}
    )
    
    respond = LLMNode(
        id="respond",
        agent=agent,
        prompt="""Based on this step-by-step analysis:

        Previous Analysis: {think[response]}
        
        Provide a clear, final response that:
        1. Addresses the core question/intent
        2. Incorporates key insights from the analysis
        3. Maintains AUG_E's personality and style
        
        Remember to keep it casual and friendly for simple greetings!""",
        next_nodes={"default": None}
    )
    
    # Create and validate graph
    graph = Graph()
    graph.add_node(think)
    graph.add_node(respond)
    
    # Add entry point
    graph.add_entry_point("main", "think")
    
    return graph

async def run_chatbot():
    """Run the reflective chatbot."""
    logger.info("Starting reflective chatbot...")
    
    graph = await create_reflection_workflow()
    
    print("\nReflective Chatbot: yo fam! i'm here to help ya think things through. what's on your mind? 🤔")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nReflective Chatbot: peace out! stay curious! ✌️")
                break
                
            state = NodeState(
                context=NodeContext(),
                results={"input": {"message": user_input}},
                data={}  # ThinkingNode will initialize step tracking
            )
            
            final_state = await graph.run("main", state)
            
            response = final_state.results.get('respond', {}).get('response', 
                "hmm, let me think about that some more...")
            print(f"\nReflective Chatbot: {response}")
            
        except KeyboardInterrupt:
            print("\nReflective Chatbot: catch ya later! ✌️")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            print("\nReflective Chatbot: oops, my brain glitched! can you try that again? 😅")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_chatbot())
