"""Tool foundation node type for the graph system.

This module provides the foundation for all tool-executing nodes in the graph system.
It handles core tool execution logic, validation, and state management.
"""

import os
import sys
import asyncio
import logging
from typing import Any, Dict, Optional, Type
from pydantic import Field, BaseModel, ValidationError

# Add parent directories to path if running directly
if __name__ == "__main__" and __package__ is None:
    file = os.path.abspath(__file__)
    parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
    sys.path.insert(0, parent)

from alchemist.ai.graph.base import Node, NodeState
from mirascope.core import BaseTool

# Configure logging
logger = logging.getLogger(__name__)

class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass

class ToolNode(Node):
    """Foundation class for nodes that execute tools.
    
    This node type provides the core functionality for tool execution within the graph.
    It handles tool initialization, validation, execution, and result management.
    
    Attributes:
        tool: The tool instance to execute
        tool_args: Dict mapping argument names to state value templates
        result_key: Key to store tool results (defaults to 'result')
        validate_args: Whether to validate tool arguments (defaults to True)
    """
    
    tool: BaseTool
    tool_args: Dict[str, str]
    result_key: str = "result"
    validate_args: bool = True
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Execute tool and manage results.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
            
        The node will:
        1. Format tool arguments from state using templates
        2. Validate arguments if enabled
        3. Execute tool with formatted arguments
        4. Store result in state under result_key
        5. Return next node ID based on success/error
        """
        try:
            # Format the tool arguments
            args = {}
            try:
                for key, template in self.tool_args.items():
                    try:
                        args[key] = template.format(**state.results)
                    except KeyError as e:
                        raise ToolExecutionError(f"Missing required state value: {e}")
                    except Exception as e:
                        raise ToolExecutionError(f"Error formatting argument {key}: {str(e)}")
            except Exception as e:
                raise ToolExecutionError(f"Error preparing tool arguments: {str(e)}")
            
            # Log the formatted arguments
            logger.debug(f"Formatted tool arguments: {args}")
            
            # Execute tool
            try:
                result = await self.tool.call(**args)
            except ValidationError as e:
                raise ToolExecutionError(f"Tool argument validation failed: {str(e)}")
            except Exception as e:
                raise ToolExecutionError(f"Tool execution failed: {str(e)}")
            
            # Store result
            state.results[self.id] = {
                self.result_key: result,
                "tool": self.tool.__class__.__name__,
                "args": args
            }
            
            logger.info(f"Tool {self.tool.__class__.__name__} executed successfully")
            return self.next_nodes.get("default")
            
        except ToolExecutionError as e:
            logger.error(f"Tool execution error in {self.id}: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "error_type": e.__class__.__name__,
                "tool": self.tool.__class__.__name__,
                "args": args if 'args' in locals() else None
            }
            return self.next_nodes.get("error")
        except Exception as e:
            logger.error(f"Unexpected error in {self.id}: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "error_type": "UnexpectedError",
                "tool": self.tool.__class__.__name__,
                "args": args if 'args' in locals() else None
            }
            return self.next_nodes.get("error")

async def test_tool_node():
    """Test tool node functionality."""
    print("\nTesting ToolNode...")
    
    # Create a mock tool for testing
    class TestTool(BaseTool):
        name: str = "test_tool"
        description: str = "A test tool"
        
        async def call(self, x: int, y: int) -> int:
            return x + y
    
    # Create test node
    node = ToolNode(
        id="test_tool",
        tool=TestTool(),
        tool_args={
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