"""Tool node for executing actions in graph workflows.

This node is intended to wrap a single function or "tool" for straightforward execution.
It does not include advanced state checks or pre/post-execution hooks. For more complex
workflow logic, see ActionNode.

Typical Use-Cases:
-----------------
1. Simple tool invocation with known inputs, e.g., arithmetic or search.
2. Quick data transformations that do not require specialized state handling.
3. Standalone function calls that do not depend on a chain of actions.

Example:
--------
>>> async def simple_adder(x: int, y: int) -> int:
...     return x + y
...
>>> node = ToolNode(
...     id="adder",
...     tool=simple_adder,
...     input_map={"x": "amount1", "y": "amount2"},
...     next_nodes={"default": "next_node", "error": "error_node"}
... )
>>> # Then pass NodeState with results/data to node.process(...)

"""

from typing import Dict, Any, Optional, Callable
from pydantic import Field
import asyncio
from alchemist.ai.base.logging import get_logger, LogComponent
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.state import NodeState, NodeStatus

# Get logger for node operations
logger = get_logger(LogComponent.NODES)

class ToolNode(Node):
    """
    Node that executes a tool or function.
    
    Differences vs. ActionNode:
        - ToolNode: Meant for direct function execution without extra checks.
        - ActionNode: Adds workflow-specific checks, pre/post hooks, chaining, etc.

    Attributes:
        tool: Callable that will be executed when this node runs. It can be async or sync.
        input_map: A mapping of tool parameter names to state keys. The state can supply
            either .data[...] or .results[...] values when used as parameters.
        output_key: The key under which the tool's return value will be stored in
            state.results[node.id].
    """
    
    tool: Callable
    input_map: Dict[str, str] = Field(default_factory=dict)
    output_key: str = "result"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Execute the tool with mapped inputs from the node state.
        
        Workflow:
            1. Gather parameters from state.data or state.results using input_map.
            2. Call the tool (async or sync).
            3. Store the result in state.results under [self.id][self.output_key].
            4. Return the ID of the next node or 'error' transition if an exception occurs.

        Args:
            state: Current node state.

        Returns:
            Optional[str]: The ID of the next node to execute.
        """
        try:
            # Let the base Node handle dotted-key retrieval
            inputs = self._prepare_input_data(state)

            # Call the tool
            if asyncio.iscoroutinefunction(self.tool):
                result = await self.tool(**inputs)
            else:
                result = self.tool(**inputs)
            
            # Store the result
            state.results[self.id] = {self.output_key: result}
            
            # Mark node complete
            state.mark_status(self.id, NodeStatus.COMPLETED)
            return self.get_next_node()
            
        except Exception as e:
            logger.error(f"Error in tool node {self.id}: {str(e)}")
            state.errors[self.id] = str(e)
            state.mark_status(self.id, NodeStatus.ERROR)
            return self.get_next_node("error")
    
    def validate(self) -> bool:
        """Validate that the tool is callable and the parent Node checks pass."""
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