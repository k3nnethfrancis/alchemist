"""Tests for ToolNode functionality with input_map and nested keys."""

import pytest
from alchemist.ai.graph.nodes.tool import ToolNode
from alchemist.ai.graph.state import NodeState, NodeStatus

@pytest.mark.asyncio
async def test_tool_node_with_input_map():
    """Test ToolNode execution with input_map and nested keys."""
    async def test_tool(x: int, y: int) -> int:
        return x * y

    node = ToolNode(
        id="test_tool",
        tool=test_tool,
        input_map={
            "x": "inputs.values.a",
            "y": "inputs.values.b"
        }
    )

    state = NodeState()
    state.set_data("inputs", {"values": {"a": 5, "b": 4}})

    next_id = await node.process(state)
    assert state.results["test_tool"]["result"] == 20
    assert state.status["test_tool"] == NodeStatus.COMPLETED
    assert next_id == node.get_next_node()
