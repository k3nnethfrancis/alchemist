"""Action node implementations for workflow tool execution.

This module provides specialized nodes for executing tools within workflows,
adding features like pre/post execution hooks, action chaining, and enhanced
state management.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pydantic import Field

# Add parent directories to path if running directly
if __name__ == "__main__" and __package__ is None:
    file = os.path.abspath(__file__)
    parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))
    sys.path.insert(0, parent)

from alchemist.ai.graph.nodes.base import ToolNode
from alchemist.ai.graph.base import NodeState
from alchemist.ai.base.tools import CalculatorTool

logger = logging.getLogger(__name__)

# Re-export ToolNode for backward compatibility
__all__ = ['ActionNode']

class ActionNode(ToolNode):
    """Specialized node for workflow actions.
    
    This node type extends ToolNode to add workflow-specific functionality:
    - Pre/post execution hooks
    - Action chaining
    - Enhanced state management
    - Result formatting
    
    Attributes:
        name: Human-readable name for the action
        description: What this action does
        chain_actions: Whether to chain with next action
        required_state: List of state keys required for execution
        preserve_state: List of state keys to preserve after execution
    """
    
    name: str
    description: str = ""
    chain_actions: bool = False
    required_state: List[str] = Field(default_factory=list)
    preserve_state: List[str] = Field(default_factory=list)
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Execute action with pre/post hooks.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
            
        The node will:
        1. Validate required state
        2. Run pre-execution hook
        3. Execute tool using parent implementation
        4. Run post-execution hook
        5. Clean up state if needed
        6. Return next node ID or chained action
        """
        try:
            # Validate required state
            self._validate_required_state(state)
            
            # Pre-execution hook
            await self.pre_execute(state)
            
            # Execute tool using parent implementation
            next_id = await super().process(state)
            
            # Post-execution hook
            await self.post_execute(state)
            
            # Clean up state
            self._cleanup_state(state)
            
            # Handle action chaining
            if self.chain_actions and next_id:
                logger.info(f"Chaining action {self.id} to {next_id}")
                state.results[self.id]["chained"] = True
            
            return next_id
            
        except Exception as e:
            logger.error(f"Error in action {self.name}: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "action": self.name,
                "state": {k: state.results.get(k) for k in self.required_state}
            }
            return self.next_nodes.get("error")
    
    async def pre_execute(self, state: NodeState) -> None:
        """Hook for setup before action execution.
        
        Args:
            state: Current node state
        """
        logger.debug(f"Pre-execute hook for action {self.name}")
    
    async def post_execute(self, state: NodeState) -> None:
        """Hook for cleanup after action execution.
        
        Args:
            state: Current node state
        """
        logger.debug(f"Post-execute hook for action {self.name}")
    
    def _validate_required_state(self, state: NodeState) -> None:
        """Ensure all required state keys are present.
        
        Args:
            state: Current node state
            
        Raises:
            ValueError: If required state is missing
        """
        missing = [k for k in self.required_state if k not in state.results]
        if missing:
            raise ValueError(f"Missing required state keys: {missing}")
    
    def _cleanup_state(self, state: NodeState) -> None:
        """Remove unnecessary state data.
        
        Only keeps state keys listed in preserve_state.
        
        Args:
            state: Current node state
        """
        if not self.preserve_state:
            return
            
        keys = list(state.results.keys())
        for k in keys:
            if k not in self.preserve_state and k != self.id:
                del state.results[k]

async def test_action_node():
    """Test action node functionality."""
    print("\nTesting ActionNode...")
    
    # Create a test calculator action
    calc = CalculatorTool()
    node = ActionNode(
        id="test_calc",
        name="Calculator Action",
        description="Adds two numbers",
        tool=calc,
        args_key="calc_args",
        required_state=["calc_args"],
        next_nodes={"default": "next", "error": "error"}
    )
    
    # Create test state
    state = NodeState()
    state.results["calc_args"] = {"expression": "2 + 2"}
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next", f"Expected 'next', got {next_id}"
    assert state.results["test_calc"]["result"] == 4, "Incorrect calculation"
    print("ActionNode test passed!")

if __name__ == "__main__":
    print(f"Running test from: {__file__}")
    asyncio.run(test_action_node()) 