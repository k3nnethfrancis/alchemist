"""Tests for graph state management."""

import pytest
from alchemist.ai.graph.base import NodeState, NodeContext
import asyncio

def test_state_initialization():
    """Test state initialization."""
    state = NodeState(
        context=NodeContext(),
        results={},
        data={}
    )
    
    assert state.context is not None
    assert state.results == {}
    assert state.data == {}

def test_state_updates():
    """Test state update operations."""
    state = NodeState(
        context=NodeContext(),
        results={},
        data={}
    )
    
    # Test results updates
    state.results["test"] = {"response": "value"}
    assert state.results["test"]["response"] == "value"
    
    # Test data updates
    state.data["counter"] = 1
    state.data["counter"] += 1
    assert state.data["counter"] == 2

def test_state_access():
    """Test state access patterns."""
    state = NodeState(
        context=NodeContext(),
        results={"node1": {"response": "test"}},
        data={"key": "value"}
    )
    
    # Test dict-style access
    assert state.results.get("node1", {}).get("response") == "test"
    assert state.data.get("key") == "value"
    
    # Test missing values
    assert state.results.get("missing", {}).get("response") is None
    assert state.data.get("missing") is None 

@pytest.mark.asyncio
async def test_async_state_updates():
    """Test async state update operations."""
    state = NodeState(
        context=NodeContext(),
        results={},
        data={}
    )
    
    # Test async updates
    state.data["async_counter"] = 0
    for i in range(3):
        state.data["async_counter"] += 1
        await asyncio.sleep(0)
    
    assert state.data["async_counter"] == 3

@pytest.mark.asyncio
async def test_state_persistence():
    """Test state persistence across async operations."""
    state = NodeState(
        context=NodeContext(),
        results={"initial": {"value": "test"}},
        data={"counter": 0}
    )
    
    # Simulate async node processing
    async def process_node():
        state.data["counter"] += 1
        state.results["node"] = {"processed": True}
        await asyncio.sleep(0)
        
    await process_node()
    
    assert state.data["counter"] == 1
    assert state.results["initial"]["value"] == "test"
    assert state.results["node"]["processed"] is True 