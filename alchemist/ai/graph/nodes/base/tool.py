"""Tool foundation node type."""

import os
import sys
import asyncio
from typing import Any, Dict, Optional
from pydantic import Field, BaseModel

# Add parent directories to path if running directly
if __name__ == "__main__" and __package__ is None:
    file = os.path.abspath(__file__)
    parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
    sys.path.insert(0, parent)

from alchemist.ai.graph.base import Node, NodeState
from alchemist.ai.base.tools import BaseTool

class MockTool(BaseTool, BaseModel):
    """Mock tool for testing."""
    name: str = "mock_tool"
    args: Dict[str, Any] = {}
    
    async def execute(self, **kwargs) -> str:
        """Return a mock result."""
        return f"Mock result for {self.name} with args {kwargs}"

class ToolNode(Node):
    """Base class for nodes that execute tools.
    
    This node type adds tool execution functionality on top of the base Node class.
    It handles tool setup, input preparation, and result storage.
    
    Attributes:
        tool_name: Name of the tool to execute
        tool_args: Arguments to pass to the tool
    """
    
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process node by executing tool.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
            
        The node will:
        1. Create tool instance
        2. Execute tool with arguments
        3. Store result
        4. Return next node ID based on success/error
        """
        try:
            # Create and execute tool
            tool = MockTool(name=self.tool_name)  # Use mock tool for testing
            result = await tool.execute(**self.tool_args)
            
            # Store result
            state.results[self.id] = {"result": result}
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            state.results[self.id] = {"error": str(e)}
            return self.next_nodes.get("error")

async def test_tool_node():
    """Test tool node functionality."""
    print("\nTesting ToolNode...")
    
    # Create a test calculator tool node
    node = ToolNode(
        id="test_tool",
        tool_name="calculator",
        tool_args={"operation": "add", "numbers": [1, 2, 3]},
        next_nodes={"default": "next_node", "error": "error_node"}
    )
    
    # Create test state
    state = NodeState()
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next_node", f"Expected 'next_node', got {next_id}"
    assert "result" in state.results["test_tool"], "No result in results"
    print(f"Tool Result: {state.results['test_tool']['result']}")
    print("ToolNode test passed!")

if __name__ == "__main__":
    print(f"Running test from: {__file__}")
    asyncio.run(test_tool_node()) 