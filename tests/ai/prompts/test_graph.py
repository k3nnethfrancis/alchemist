"""Graph Prompt Tests

Tests the generation of prompts for different node types in the graph system.
"""

import pytest
from typing import Dict, Any

from alchemist.ai.prompts.graph import (
    create_decision_prompt,
    create_action_prompt,
    create_response_prompt,
    GraphPromptConfig,
    NodeType
)
from alchemist.ai.prompts.persona import AUG_E

@pytest.fixture
def persona():
    """Test persona fixture."""
    return AUG_E

@pytest.fixture
def graph_state():
    """Sample graph state for testing."""
    return {
        "message": "Hello there!",
        "history": [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ],
        "current_node": "test_node",
        "variables": {"test_var": "test_value"}
    }

def test_decision_prompt_generation(persona):
    """Test generating decision node prompts."""
    config = GraphPromptConfig(
        node_type=NodeType.DECISION,
        options=["option_a", "option_b"]
    )
    
    prompt = create_decision_prompt(persona, config)
    
    # Verify prompt structure
    assert isinstance(prompt, Dict)
    assert "role" in prompt
    assert "content" in prompt
    
    # Verify content includes key elements
    content = prompt["content"]
    assert persona.name in content
    assert "decision" in content.lower()
    assert all(option in content for option in config.options)

def test_action_prompt_generation(persona, graph_state):
    """Test generating action node prompts."""
    config = GraphPromptConfig(
        node_type=NodeType.ACTION,
        tools=["tool_a", "tool_b"]
    )
    
    prompt = create_action_prompt(persona, config, graph_state)
    
    # Verify prompt includes state context
    content = prompt["content"]
    assert "message" in content
    assert "history" in content
    assert all(tool in content for tool in config.tools)
    assert "variables" in content
    assert "test_value" in content

def test_response_prompt_generation(persona, graph_state):
    """Test generating response node prompts."""
    config = GraphPromptConfig(
        node_type=NodeType.RESPONSE,
        style_guidelines=["be_concise", "stay_in_character"]
    )
    
    prompt = create_response_prompt(persona, config, graph_state)
    
    # Verify prompt maintains character and includes guidelines
    content = prompt["content"]
    assert persona.name in content
    assert all(guideline in content for guideline in config.style_guidelines)
    assert "message" in content
    assert "history" in content 