"""Context supplier nodes for injecting dynamic information into the graph.

These nodes act as bridges between external systems and the graph workflow,
providing real-time context, facts, and state information. They are inspired by
Eliza's Provider pattern.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field

from alchemist.ai.graph.base import Node, NodeState
from alchemist.ai.base.agent import BaseAgent

logger = logging.getLogger(__name__)

class ContextSupplierNode(Node):
    """Base class for nodes that supply dynamic context to the graph.
    
    Context suppliers inject real-time information, facts, and state into
    the graph workflow. They maintain consistent data access patterns and
    format information appropriately for consumption by other nodes.
    
    Attributes:
        target_key: Key in state where context will be stored
        cache_ttl: Optional TTL for cached data in seconds
        required_context: List of required context keys
    """
    
    target_key: str
    cache_ttl: Optional[int] = None
    required_context: List[str] = Field(default_factory=list)

    async def process(self, state: NodeState) -> Optional[str]:
        """Process the current state and inject context.
        
        Args:
            state: Current node state
            
        Returns:
            ID of next node to execute or None if finished
        """
        try:
            # Validate required context
            for key in self.required_context:
                if key not in state.context.metadata:
                    logger.error(f"Missing required context: {key}")
                    return self.next_nodes.get("error")
            
            # Get context data
            context_data = await self.get_context(state)
            
            # Store in state
            state.context.metadata[self.target_key] = context_data
            
            # Record results
            state.results[self.id] = {
                "context_key": self.target_key,
                "context_type": self.__class__.__name__
            }
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            logger.error(f"Error in context supplier: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            return self.next_nodes.get("error")

    async def get_context(self, state: NodeState) -> Dict[str, Any]:
        """Get context data. Override in subclasses."""
        raise NotImplementedError()

class TimeContextNode(ContextSupplierNode):
    """Node that provides temporal context.
    
    Supplies current time, date, and other temporal information to the workflow.
    Can be configured to provide different time formats and granularity.
    """
    
    time_format: str = "%Y-%m-%d %H:%M:%S"
    include_timezone: bool = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.target_key:
            self.target_key = "temporal_context"
    
    async def get_context(self, state: NodeState) -> Dict[str, Any]:
        now = datetime.now()
        
        context = {
            "current_time": now.strftime(self.time_format),
            "timestamp": now.timestamp(),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "weekday": now.strftime("%A")
        }
        
        if self.include_timezone:
            context["timezone"] = datetime.now().astimezone().tzname()
            
        return context

class FactsContextNode(ContextSupplierNode):
    """Node that provides relevant facts and memory context.
    
    Maintains and supplies conversation facts, memory entries, and other
    contextual information that helps maintain conversation coherence.
    """
    
    max_facts: int = 10
    embedding_key: str = "content"
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.target_key:
            self.target_key = "facts_context"
    
    async def get_context(self, state: NodeState) -> Dict[str, Any]:
        # Get recent messages for context
        recent_messages = []
        for result in state.results.values():
            if "response" in result:
                recent_messages.append(result["response"])
        
        # Get relevant facts from memory
        facts = state.context.memory.get("facts", [])
        if facts:
            facts = facts[-self.max_facts:]  # Get most recent facts
            
        return {
            "recent_messages": recent_messages,
            "known_facts": facts,
            "fact_count": len(facts)
        }

class EngagementContextNode(ContextSupplierNode):
    """Node that tracks conversation engagement and dynamics.
    
    Similar to Eliza's Boredom Provider, this node monitors conversation
    flow and engagement levels, helping guide response generation.
    """
    
    class EngagementLevel(BaseModel):
        """Model for engagement level configuration."""
        min_score: float
        status_messages: List[str]
        
    engagement_levels: List[EngagementLevel] = Field(default_factory=list)
    lookback_messages: int = 10
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.target_key:
            self.target_key = "engagement_context"
            
        # Default engagement levels if none provided
        if not self.engagement_levels:
            self.engagement_levels = [
                EngagementLevel(
                    min_score=0.8,
                    status_messages=["Highly engaged", "Active participation"]
                ),
                EngagementLevel(
                    min_score=0.5,
                    status_messages=["Moderately engaged", "Normal interaction"]
                ),
                EngagementLevel(
                    min_score=0.0,
                    status_messages=["Low engagement", "Minimal interaction"]
                )
            ]
    
    async def get_context(self, state: NodeState) -> Dict[str, Any]:
        # Analyze recent messages
        recent_messages = []
        for result in state.results.values():
            if "response" in result:
                recent_messages.append(result["response"])
        
        # Calculate engagement score
        score = await self._calculate_engagement(recent_messages)
        
        # Get appropriate engagement level
        level = None
        for eng_level in sorted(
            self.engagement_levels,
            key=lambda x: x.min_score,
            reverse=True
        ):
            if score >= eng_level.min_score:
                level = eng_level
                break
                
        return {
            "engagement_score": score,
            "message_count": len(recent_messages),
            "status": level.status_messages[0] if level else "Unknown",
            "last_message_time": datetime.now().isoformat()
        }
    
    async def _calculate_engagement(self, messages: List[str]) -> float:
        """Calculate engagement score from messages."""
        if not messages:
            return 0.0
            
        # Simple scoring based on message frequency and length
        total_length = sum(len(m) for m in messages)
        avg_length = total_length / len(messages)
        
        # Normalize to 0-1 range
        score = min(1.0, (avg_length / 100) * (len(messages) / self.lookback_messages))
        
        return score

async def test_time_context():
    """Test time context functionality."""
    print("\nTesting TimeContextNode...")
    
    # Create test node
    node = TimeContextNode(
        id="test_time",
        target_key="time_context",
        next_nodes={"default": "next_node"}
    )
    
    # Create test state
    state = NodeState()
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next_node", "Incorrect next node"
    assert "time_context" in state.context.metadata, "Time context not in metadata"
    context = state.context.metadata["time_context"]
    assert "current_time" in context, "Missing current time"
    assert "timestamp" in context, "Missing timestamp"
    print("TimeContextNode test passed!")

async def test_facts_context():
    """Test facts context functionality."""
    print("\nTesting FactsContextNode...")
    
    # Create test node
    node = FactsContextNode(
        id="test_facts",
        target_key="facts_context",
        next_nodes={"default": "next_node"}
    )
    
    # Create test state
    state = NodeState()
    state.context.memory["facts"] = [
        "Fact 1: Test fact",
        "Fact 2: Another fact"
    ]
    state.results["previous"] = {"response": "Test response"}
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next_node", "Incorrect next node"
    assert "facts_context" in state.context.metadata, "Facts context not in metadata"
    context = state.context.metadata["facts_context"]
    assert "known_facts" in context, "Missing facts"
    assert len(context["known_facts"]) == 2, "Incorrect fact count"
    print("FactsContextNode test passed!")

async def test_engagement_context():
    """Test engagement context functionality."""
    print("\nTesting EngagementContextNode...")
    
    # Create test node
    node = EngagementContextNode(
        id="test_engagement",
        target_key="engagement_context",
        next_nodes={"default": "next_node"}
    )
    
    # Create test state
    state = NodeState()
    state.results["msg1"] = {"response": "Test response 1"}
    state.results["msg2"] = {"response": "Test response 2"}
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next_node", "Incorrect next node"
    assert "engagement_context" in state.context.metadata, "Engagement context not in metadata"
    context = state.context.metadata["engagement_context"]
    assert "engagement_score" in context, "Missing engagement score"
    assert "status" in context, "Missing status"
    print("EngagementContextNode test passed!")

async def run_all_tests():
    """Run all context supplier tests."""
    print("Running context supplier tests...")
    await test_time_context()
    await test_facts_context()
    await test_engagement_context()
    print("\nAll context supplier tests passed!")

if __name__ == "__main__":
    print(f"Running tests from: {__file__}")
    import asyncio
    asyncio.run(run_all_tests()) 