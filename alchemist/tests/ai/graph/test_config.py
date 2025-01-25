"""Tests for graph configuration."""

import pytest
from alchemist.ai.graph.config import GraphConfig

def test_graph_config_initialization():
    """Test GraphConfig initialization and basic operations."""
    config = GraphConfig()
    assert config.config == {}
    assert config.node_configs == {}

def test_graph_config_operations():
    """Test GraphConfig data operations."""
    config = GraphConfig()
    config.set("key1", "value1")
    assert config.get("key1") == "value1"

    node_config = {"param1": "value1"}
    config.set_node_config("NodeType1", node_config)
    assert config.get_node_config("NodeType1") == node_config

def test_graph_config_from_dict():
    """Test GraphConfig creation from dictionary."""
    input_config = {
        "key1": "value1",
        "key2": {"nested": "value"}
    }
    config = GraphConfig.from_dict(input_config)
    assert config.get("key1") == "value1"
    assert config.get("key2")["nested"] == "value"
