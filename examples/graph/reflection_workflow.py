"""
A simple reflective chatbot using the Graph framework.

This example demonstrates how to:
1. Create a graph with two nodes (thinking and response)
2. Use LLM nodes for reflection and response generation
3. Run an interactive chat loop with the workflow
"""

import asyncio
from pathlib import Path
from alchemist.ai.graph.base import Graph, NodeState
from alchemist.ai.graph.nodes.base.llm import LLMNode
from alchemist.ai.base.logging import configure_logging, LogComponent, LogLevel

# Set up basic logging
log_dir = Path("logs/reflection_workflow")
log_dir.mkdir(parents=True, exist_ok=True)
configure_logging(log_file=str(log_dir / "reflection.log"))

class ReflectionWorkflow:
    """A simple workflow that reflects on user input and generates thoughtful responses."""
    
    def __init__(self):
        self.graph = self._create_graph()
    
    def _create_graph(self) -> Graph:
        """Create a two-node graph for reflection and response."""
        graph = Graph()
        
        # First node: Think deeply about the user's input
        think_node = LLMNode(
            id="think",
            prompt="""Reflect deeply on this input, considering:
            - The underlying meaning or intent
            - Any assumptions or implications
            - Potential areas for exploration
            
            User input: {message}""",
            next_nodes={"default": "respond"}
        )
        
        # Second node: Generate a friendly, insightful response
        respond_node = LLMNode(
            id="respond",
            prompt="""Based on our reflection, craft a friendly and insightful response.
            Make it conversational but meaningful.
            
            Our reflection: {think_result}""",
            next_nodes={"default": None}  # End of workflow
        )
        
        # Add nodes and entry point
        graph.add_node(think_node)
        graph.add_node(respond_node)
        graph.add_entry_point("start", "think")
        
        return graph
    
    async def chat(self):
        """Run an interactive chat session."""
        print("\nReflective Bot: Hi! I'm here to help you reflect. Type 'exit' to quit.")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nReflective Bot: Goodbye! Take care!")
                    break
                
                # Prepare state with user's message
                state = NodeState()
                state.data["message"] = user_input
                
                # Run the reflection workflow
                final_state = await self.graph.run("start", state)
                
                # Get the response from the final node
                response = final_state.results.get("respond", {}).get("response", 
                    "I need a moment to reflect on that.")
                print(f"\nReflective Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nReflective Bot: Session ended. Take care!")
                break
            except Exception as e:
                print(f"\nReflective Bot: I encountered an error: {str(e)}")

def main():
    """Run the reflection workflow."""
    workflow = ReflectionWorkflow()
    asyncio.run(workflow.chat())

if __name__ == "__main__":
    main()
