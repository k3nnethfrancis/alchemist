"""Action node implementations for tool execution."""

from typing import Dict, Any, Optional
from pydantic import Field

from alchemist.ai.graph.nodes.base.tool import ToolNode

# Re-export ToolNode for backward compatibility
__all__ = ['ToolNode']

class ToolNode(Node):
    """Node for executing specific tools.
    
    This node type handles tool execution within the graph. It can be configured
    with a tool name and arguments, and will execute the tool during processing.
    
    Attributes:
        tool_name: Name of the tool to execute
        tool_args: Dictionary of arguments to pass to the tool
        parallel: Whether this node can run in parallel (default True for tools)
    """
    
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    parallel: bool = True  # Tools can typically run in parallel
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Execute tool and store result.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
            
        The node will:
        1. Execute the specified tool with provided arguments
        2. Store the tool result in the state
        3. Return the next node ID based on success/error
        """
        try:
            # TODO: Replace mock with actual tool execution
            # For now, just simulate tool execution
            result = {
                "tool": self.tool_name,
                "args": self.tool_args,
                "data": f"Sample data from {self.tool_name}",
                "metadata": {"executed_at": state.context.get_context("time")}
            }
            
            # Store result
            state.results[self.id] = result
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            state.results[self.id] = {
                "error": str(e),
                "tool": self.tool_name,
                "args": self.tool_args
            }
            return self.next_nodes.get("error") 