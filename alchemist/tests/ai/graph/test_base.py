"""Tests for core Graph functionality including chain utility and looping."""

import pytest
from alchemist.ai.graph.base import Graph
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.nodes.llm import LLMNode
from alchemist.ai.graph.nodes.tool import ToolNode
from alchemist.ai.graph.nodes.terminal import TerminalNode
from alchemist.ai.graph.state import NodeState, NodeStatus
from typing import Optional

class TestNode(Node):
    """Test node that increments a counter."""
    async def process(self, state: NodeState) -> Optional[str]:
        count = state.data.get("count", 0)
        state.data["count"] = count + 1
        state.set_result(self.id, "count", state.data["count"])
        return self.get_next_node()

@pytest.mark.asyncio
async def test_graph_chain_utility():
    """Test the chain utility to sequentially link nodes."""
    graph = Graph()

    node1 = TestNode(id="node1")
    node2 = TestNode(id="node2")
    end_node = TerminalNode(id="end")

    graph.chain([node1, node2, end_node])
    graph.add_entry_point("start", "node1")

    errors = graph.validate()
    assert not errors, f"Graph validation failed: {errors}"

    state = NodeState()
    final_state = await graph.run("start", state)

    assert final_state.results["node1"]["count"] == 1
    assert final_state.results["node2"]["count"] == 2
    assert final_state.status["end"] == NodeStatus.TERMINAL

@pytest.mark.asyncio
async def test_graph_looping():
    """Test graph execution with looping."""
    graph = Graph()

    class LoopNode(Node):
        async def process(self, state: NodeState) -> Optional[str]:
            count = state.data.get("loop_count", 0)
            state.data["loop_count"] = count + 1
            if count < 2:
                return self.id  # Loop back to this node
            return self.get_next_node()

    loop_node = LoopNode(id="loop_node")
    end_node = TerminalNode(id="end")

    loop_node.next_nodes["default"] = "end"

    graph.add_node(loop_node)
    graph.add_node(end_node)
    graph.add_entry_point("start", "loop_node")

    errors = graph.validate()
    assert not errors, f"Graph validation failed: {errors}"

    state = NodeState()
    final_state = await graph.run("start", state)

    assert state.data["loop_count"] == 3  # Loop runs 3 times
    assert final_state.status["end"] == NodeStatus.TERMINAL
