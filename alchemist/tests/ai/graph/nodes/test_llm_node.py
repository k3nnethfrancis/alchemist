"""Tests for LLMNode functionality with input_map and nested keys."""

import pytest
from alchemist.ai.graph.nodes.llm import LLMNode
from alchemist.ai.graph.state import NodeState, NodeStatus
from mirascope.core import Messages, prompt_template

@pytest.mark.asyncio
async def test_llm_node_with_input_map():
    """Test LLMNode with input_map and nested keys."""
    @prompt_template()
    def test_prompt(name: str, info: str) -> Messages.Type:
        return Messages.User(f"Hello {name}, here is your info: {info}")

    node = LLMNode(
        id="test_llm",
        prompt_template=test_prompt,
        input_map={
            "name": "user.profile.name",
            "info": "data.info"
        },
        system_prompt="You are a helpful assistant."
    )

    state = NodeState()
    state.set_data("user", {"profile": {"name": "Alice"}})
    state.set_data("data", {"info": "Important information"})

    next_id = await node.process(state)
    assert "response" in state.results["test_llm"]
    assert state.results["test_llm"]["response"] is not None
    assert state.status["test_llm"] == NodeStatus.COMPLETED
    assert next_id == node.get_next_node()
