"""Action node implementations for workflow tool execution.

In contrast to ToolNode, this module provides specialized nodes for executing tools
in more complex workflows. ActionNode supports:
    - Required state checks (ensuring certain data is present)
    - Pre/post execution hooks (e.g., logging, retries)
    - Action chaining
    - Optional state cleaning or preservation

ActionNode is ideal if you need more control over the lifecycle of a single tool call.

Example:
--------
>>> from alchemist.ai.base.tools import CalculatorTool
>>>
>>> node = ActionNode(
...     id="calc_step",
...     name="Calculator",
...     description="Adds two numbers from node state",
...     tool=CalculatorTool(),
...     required_state=["calc_args"],
...     preserve_state=["calc_args", "calc_step"],  # Keep these results
...     next_nodes={"default": "next_node", "error": "error_node"},
... )
>>> # The node ensures 'calc_args' is present in the state before calling.
>>> # After execution, it removes extraneous keys from state.results.
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

from alchemist.ai.graph.nodes.tool import ToolNode
from alchemist.ai.graph.base import NodeState
from alchemist.ai.tools.calculator import CalculatorTool

logger = logging.getLogger(__name__)

# Re-export ToolNode for backward compatibility
__all__ = ['ActionNode']

class ActionNode(ToolNode):
    """
    Specialized node for workflow actions, extending ToolNode with workflow logic.

    Additional Features vs. ToolNode:
    ---------------------------------
    - required_state: A list of keys to verify before execution (fails if missing).
    - preserve_state: A list of keys to keep in NodeState.results after execution. Other
      entries may be removed to avoid polluting subsequent steps.
    - name: Human-readable name for the action.
    - description: High-level description (metadata) of the action.

    Attributes:
        name (str): A human-readable name for the action.
        description (str): A description of what this action does.
        chain_actions (bool): Whether to automatically chain the next action if one is set.
        required_state (List[str]): NodeState keys that must be present in results to proceed.
        preserve_state (List[str]): NodeState keys that must remain after execution.

    Usage:
    ------
    - Pre-execution hook: `_validate_required_state(...)`
    - Execute the tool: inherited from ToolNode
    - Post-execution hook: `_cleanup_state(...)`
    """

    name: str
    description: str = ""
    chain_actions: bool = False
    required_state: List[str] = Field(default_factory=list)
    preserve_state: List[str] = Field(default_factory=list)
    
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Execute action with optional pre/post hooks.

        Workflow:
            1. Validate required state keys.
            2. Call the underlying tool (ToolNode process).
            3. Clean up state if preserve_state is specified.
            4. Return next node or error path.

        Args:
            state: Current node state containing results and data.

        Returns:
            str: The ID of the next node or None if this node is terminal.
        """
        try:
            # 1. Validate presence of required keys in data/results
            self._validate_required_state(state)
            
            # 2. Run the inherited ToolNode logic
            next_node_id = await super().process(state)
            
            # 3. Clean up after execution
            self._cleanup_state(state)
            
            # 4. Possibly chain next node
            if self.chain_actions and next_node_id:
                logger.info(f"Chaining action {self.id} to {next_node_id}")
                state.results[self.id]["chained"] = True
            
            return next_node_id
            
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
        """
        Required keys must exist in either state.results or state.data.
        """
        missing = []
        for key in self.required_state:
            if key not in state.results and key not in state.data:
                missing.append(key)

        if missing:
            raise ValueError(f"Missing required state keys: {missing}")
    
    def _cleanup_state(self, state: NodeState) -> None:
        """
        Manage state preservation:
        1. Copy preserved values from state.data to state.results if they exist
        2. Keep preserved values already in state.results
        3. Remove non-preserved values from state.results
        """
        if not self.preserve_state:
            return

        # First copy preserved values from state.data to results
        for key in self.preserve_state:
            if key in state.data and key not in state.results:
                state.results[key] = state.data[key]

        # Then remove non-preserved values from results
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
        preserve_state=["calc_args", "calc_step"],
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