"""Tests for graph state management."""

import pytest
from datetime import datetime
from alchemist.ai.graph.state import NodeState, NodeStatus, StateManager

def test_node_state_initialization():
    """Test NodeState initialization and basic properties."""
    state = NodeState()
    assert state.results == {}
    assert state.data == {}
    assert state.errors == {}
    assert state.status == {}
    assert isinstance(state.created_at, datetime)
    assert isinstance(state.updated_at, datetime)

def test_node_state_nested_data_access():
    """Test nested key access in NodeState data."""
    state = NodeState()
    state.set_data("user", {"profile": {"name": "Alice"}})
    value = state.get_nested_data("user.profile.name")
    assert value == "Alice"

def test_node_state_nested_result_access():
    """Test nested key access in NodeState results."""
    state = NodeState()
    state.set_result("node1", "output", {"details": {"value": 42}})
    value = state.get_nested_result("node1", "output.details.value")
    assert value == 42

def test_node_state_nested_key_error():
    """Test error handling for invalid nested keys."""
    state = NodeState()
    state.set_data("user", {"profile": {"name": "Alice"}})
    with pytest.raises(ValueError):
        state.get_nested_data("user.profile.age")

def test_node_state_operations():
    """Test NodeState data operations."""
    state = NodeState()
    
    # Test data operations
    state.set_data("key", "value")
    assert state.get_data("key") == "value"
    
    # Test result operations
    state.set_result("node1", "output", "result")
    assert state.get_result("node1", "output") == "result"
    
    # Test status operations
    state.mark_status("node1", NodeStatus.RUNNING)
    assert state.status["node1"] == NodeStatus.RUNNING

def test_state_manager():
    """Test StateManager operations."""
    manager = StateManager()
    
    # Test state creation and persistence
    state = manager.create_state()
    state.set_data("test", "value")
    
    manager.persist_state("key1", state)
    retrieved = manager.retrieve_state("key1")
    
    assert retrieved is not None
    assert retrieved.get_data("test") == "value"
