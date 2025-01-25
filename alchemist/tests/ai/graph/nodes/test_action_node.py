"""Tests for ActionNode functionality."""

import pytest
from alchemist.ai.graph.nodes.actions import ActionNode
from alchemist.ai.graph.state import NodeState, NodeStatus

@pytest.mark.asyncio
async def test_action_node():
    """Test ActionNode with required state and preservation."""
    async def test_tool(x: int) -> int:
        return x * 2

    node = ActionNode(
        id="test_action",
        name="Test Action",
        description="Test multiplication",
        tool=test_tool,
        required_state=["input_value"],
        preserve_state=["input_value", "test_action"],
        input_map={"x": "input_value"}
    )

    state = NodeState()
    state.set_data("input_value", 5)

    next_id = await node.process(state)
    
    assert state.results["test_action"]["result"] == 10
    assert "input_value" in state.results  # Should be preserved
    assert state.status["test_action"] == NodeStatus.COMPLETED
