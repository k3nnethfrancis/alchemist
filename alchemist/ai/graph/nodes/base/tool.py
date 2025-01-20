"""Tool foundation node type for the graph system.

This module provides the foundation for all tool-executing nodes in the graph system.
It handles core tool execution logic, validation, and state management.
"""

import os
import sys
import asyncio
from typing import Any, Dict, Optional, Type
from pydantic import Field, BaseModel

# Add parent directories to path if running directly
if __name__ == "__main__" and __package__ is None:
    file = os.path.abspath(__file__)
    parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
    sys.path.insert(0, parent)

from alchemist.ai.graph.base import Node, NodeState
from alchemist.ai.base.tools import BaseTool

class ToolNode(Node):
    """Foundation class for nodes that execute tools.
    
    This node type provides the core functionality for tool execution within the graph.
    It handles tool initialization, validation, execution, and result management.
    
    Attributes:
        tool: The tool instance to execute
        args_key: Key in state to find tool arguments (optional)
        result_key: Key to store tool results (defaults to 'result')
        validate_args: Whether to validate tool arguments (defaults to True)
    """
    
    tool: BaseTool
    args_key: Optional[str] = None
    result_key: str = "result"
    validate_args: bool = True
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Execute tool and manage results.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
            
        The node will:
        1. Validate tool and arguments
        2. Execute tool with arguments from state if args_key provided
        3. Store result in state under result_key
        4. Return next node ID based on success/error
        """
        try:
            # Get arguments from state if key provided
            args = {}
            if self.args_key and self.args_key in state.results:
                args = state.results[self.args_key]
            
            # Validate arguments if enabled
            if self.validate_args:
                self._validate_args(args)
            
            # Execute tool
            result = await self.tool.execute(**args)
            
            # Store result
            state.results[self.id] = {self.result_key: result}
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            state.results[self.id] = {
                "error": str(e),
                "tool": self.tool.__class__.__name__,
                "args": args
            }
            return self.next_nodes.get("error")
    
    def _validate_args(self, args: Dict[str, Any]) -> None:
        """Validate tool arguments before execution.
        
        Args:
            args: Arguments to validate
            
        Raises:
            ValueError: If required arguments are missing or invalid
        """
        if not hasattr(self.tool, "validate_args"):
            return
            
        self.tool.validate_args(**args)

async def test_tool_node():
    """Test tool node functionality."""
    print("\nTesting ToolNode...")
    
    # Create a mock tool for testing
    class TestTool(BaseTool):
        async def execute(self, x: int, y: int) -> int:
            return x + y
    
    # Create test node
    node = ToolNode(
        id="test_tool",
        tool=TestTool(),
        args_key="calc_args",
        next_nodes={"default": "next", "error": "error"}
    )
    
    # Create test state
    state = NodeState()
    state.results["calc_args"] = {"x": 1, "y": 2}
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next", f"Expected 'next', got {next_id}"
    assert state.results["test_tool"]["result"] == 3, "Incorrect calculation result"
    print("ToolNode test passed!")

if __name__ == "__main__":
    print(f"Running test from: {__file__}")
    asyncio.run(test_tool_node()) 