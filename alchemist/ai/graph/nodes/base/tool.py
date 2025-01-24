"""Tool node for executing actions in graph workflows."""

from typing import Dict, Any, Optional, Callable, Union
from pydantic import Field
import asyncio
from alchemist.ai.base.logging import get_logger, LogComponent
from alchemist.ai.graph.nodes.base.node import Node, NodeState

# Get logger for node operations
logger = get_logger(LogComponent.NODES)

class ToolNode(Node):
    """
    Node that executes a tool or function.
    
    Attributes:
        tool: Function to execute
        input_map: Mapping of tool parameters to state data
        output_key: Key to store tool output in results
    """
    
    tool: Callable
    input_map: Dict[str, str] = Field(default_factory=dict)
    output_key: str = "result"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Execute tool with mapped inputs from state.
        
        Args:
            state: Current node state
            
        Returns:
            str: Next node ID
        """
        try:
            # Map inputs from state
            inputs = {}
            for param, state_key in self.input_map.items():
                # Try to get from state data first
                if state_key in state.data:
                    inputs[param] = state.data[state_key]
                    continue
                    
                # Then try results from other nodes
                for node_results in state.results.values():
                    if isinstance(node_results, dict) and state_key in node_results:
                        inputs[param] = node_results[state_key]
                        break
                        
                if param not in inputs:
                    raise ValueError(f"Could not find input '{state_key}' for parameter '{param}'")
            
            # Execute tool
            result = await self.tool(**inputs)
            
            # Store result
            state.results[self.id] = {self.output_key: result}
            
            return self.get_next_node()
            
        except Exception as e:
            logger.error(f"Error in tool node {self.id}: {str(e)}")
            state.errors[self.id] = str(e)
            return self.get_next_node("error")
    
    def validate(self) -> bool:
        """Validate node configuration."""
        if not callable(self.tool):
            return False
        return super().validate()

async def test_tool_node():
    """Test tool node functionality."""
    print("\nTesting ToolNode...")
    
    # Create a mock tool for testing
    class TestTool(Callable):
        def __call__(self, x: int, y: int) -> int:
            return x + y
    
    # Create test node
    node = ToolNode(
        id="test_tool",
        tool=TestTool(),
        input_map={
            "x": "{calc.x}",
            "y": "{calc.y}"
        },
        next_nodes={"default": "next", "error": "error"}
    )
    
    # Create test state
    state = NodeState()
    state.results["calc"] = {"x": 1, "y": 2}
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next", f"Expected 'next', got {next_id}"
    assert state.results["test_tool"]["result"] == 3, "Incorrect calculation result"
    print("ToolNode test passed!")

if __name__ == "__main__":
    print(f"Running test from: {__file__}")
    asyncio.run(test_tool_node()) 