"""Evaluator node implementations for extracting information and building state."""

import logging
from typing import Dict, Any, Optional, List, Union
from pydantic import Field
from datetime import datetime

from alchemist.ai.graph.base import Node, NodeState
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.base.agent import BaseAgent

logger = logging.getLogger(__name__)

class EvaluatorNode(Node):
    """Base class for nodes that extract information and build state.
    
    This node type is responsible for analyzing responses or state to extract
    specific information, track patterns, and build up context over time.
    It can identify facts, track goals, and contribute to memory building.
    
    Attributes:
        target_key: Key in results to evaluate
        memory_key: Optional key to store extracted info in memory
        metadata_key: Optional key to store extracted info in metadata
    """
    
    target_key: str = "response"
    memory_key: Optional[str] = None
    metadata_key: Optional[str] = None

    async def process(self, state: NodeState) -> Optional[str]:
        """Process the current state to extract information.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute or None if finished
        """
        try:
            # Get data to evaluate
            data = self._get_target_data(state)
            if not data:
                logger.error(f"No data found for key: {self.target_key}")
                return self.next_nodes.get("error")
            
            # Extract information
            extracted_info = await self.extract_information(data, state)
            
            # Store results
            state.results[self.id] = {
                "extracted": extracted_info,
                "source_key": self.target_key
            }
            
            # Update memory if configured
            if self.memory_key and extracted_info:
                if self.memory_key not in state.context.memory:
                    state.context.memory[self.memory_key] = []
                state.context.memory[self.memory_key].append(extracted_info)
            
            # Update metadata if configured
            if self.metadata_key and extracted_info:
                state.context.metadata[self.metadata_key] = extracted_info
            
            return self.next_nodes.get("default")
            
        except Exception as e:
            logger.error(f"Error in evaluator node: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "source_key": self.target_key
            }
            return self.next_nodes.get("error")

    async def extract_information(self, data: Any, state: NodeState) -> Dict[str, Any]:
        """Extract information from the data. Override in subclasses."""
        raise NotImplementedError()

    def _get_target_data(self, state: NodeState) -> Optional[Any]:
        """Get data to evaluate from state."""
        # Try to get from results first
        for node_results in state.results.values():
            if self.target_key in node_results:
                return node_results[self.target_key]
        
        # Try context metadata
        if self.target_key in state.context.metadata:
            return state.context.metadata[self.target_key]
        
        # Try context memory
        return state.context.memory.get(self.target_key)

class FactExtractorNode(LLMNode, EvaluatorNode):
    """Node that extracts factual claims and information.
    
    This node analyzes responses to identify and classify factual claims,
    opinions, and other relevant information. It can track whether facts
    are new or previously known.
    """
    
    def __init__(self, **data):
        super().__init__(**data)
        self.prompt = """Extract and classify factual claims from this content.
For each claim, indicate:
- Whether it's a fact or opinion
- If it appears in background/context
- If it's new information

Format as JSON list of objects with properties:
- claim: The actual claim
- type: "fact" or "opinion"
- in_context: true/false
- is_new: true/false

Content to analyze:
{content}"""

    async def extract_information(self, data: Any, state: NodeState) -> Dict[str, Any]:
        response = await self.agent.get_response(self.prompt.format(content=data))
        return {"facts": response}

class GoalTrackerNode(LLMNode, EvaluatorNode):
    """Node that tracks progress towards goals and objectives.
    
    This node maintains a record of goals, their status, and completion
    criteria. It updates goal status based on new information and can
    identify when objectives are met.
    """
    
    def __init__(self, **data):
        super().__init__(**data)
        self.prompt = """Track progress on goals and objectives.
Review the current state and update goal status.

Current goals and status:
{goals}

New information to consider:
{content}

Format response as JSON list of goal objects with:
- id: Goal identifier
- name: Goal description
- status: "NOT_STARTED", "IN_PROGRESS", "COMPLETED", or "BLOCKED"
- progress: Percentage complete (0-100)
- updates: List of status changes or progress notes"""

    async def extract_information(self, data: Any, state: NodeState) -> Dict[str, Any]:
        # Get current goals from memory or initialize
        current_goals = state.context.memory.get("goals", [])
        
        response = await self.agent.get_response(
            self.prompt.format(
                goals=current_goals,
                content=data
            )
        )
        return {"goals": response}

class PatternTrackerNode(LLMNode, EvaluatorNode):
    """Node that identifies and tracks patterns in responses.
    
    This node analyzes responses over time to identify recurring themes,
    behaviors, or patterns. It maintains a history of observations and
    can detect significant changes or trends.
    """
    
    def __init__(self, **data):
        super().__init__(**data)
        self.prompt = """Analyze this content for patterns and recurring themes.
Consider previous observations and identify:
- Recurring topics or themes
- Changes in tone or approach
- New or emerging patterns
- Deviations from established patterns

Previous observations:
{patterns}

New content to analyze:
{content}

Format response as JSON object with:
- themes: List of identified themes
- changes: List of notable changes
- patterns: List of active patterns
- deviations: List of pattern breaks"""

    async def extract_information(self, data: Any, state: NodeState) -> Dict[str, Any]:
        # Get previous patterns from memory
        previous_patterns = state.context.memory.get("patterns", [])
        
        response = await self.agent.get_response(
            self.prompt.format(
                patterns=previous_patterns,
                content=data
            )
        )
        return {"patterns": response}

async def test_fact_extractor():
    """Test fact extraction functionality."""
    print("\nTesting FactExtractorNode...")
    
    # Create test agent
    class TestAgent(BaseAgent):
        async def get_response(self, prompt: str) -> str:
            return """[
                {
                    "claim": "GPT-4 has improved reasoning capabilities",
                    "type": "fact",
                    "in_context": false,
                    "is_new": true
                },
                {
                    "claim": "AI will replace all jobs",
                    "type": "opinion",
                    "in_context": false,
                    "is_new": false
                }
            ]"""
        
        async def _call(self, messages: List[Dict[str, str]], **kwargs) -> str:
            return await self.get_response(messages[0]["content"])
    
    # Create test node
    node = FactExtractorNode(
        id="test_facts",
        agent=TestAgent(),
        memory_key="extracted_facts",
        next_nodes={"default": "next_node"}
    )
    
    # Create test state
    state = NodeState()
    state.results["previous"] = {
        "response": "GPT-4 shows improved reasoning capabilities. Some say AI will replace all jobs."
    }
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next_node", "Incorrect next node"
    assert "test_facts" in state.results, "Results not in state"
    assert "extracted" in state.results["test_facts"], "No extracted data"
    assert "extracted_facts" in state.context.memory, "Facts not stored in memory"
    print("FactExtractorNode test passed!")

async def test_goal_tracker():
    """Test goal tracking functionality."""
    print("\nTesting GoalTrackerNode...")
    
    # Create test agent
    class TestAgent(BaseAgent):
        async def get_response(self, prompt: str) -> str:
            return """[
                {
                    "id": "research_analysis",
                    "name": "Complete GPT-4 Research Analysis",
                    "status": "IN_PROGRESS",
                    "progress": 75,
                    "updates": ["Completed capability analysis", "Started impact assessment"]
                }
            ]"""
        
        async def _call(self, messages: List[Dict[str, str]], **kwargs) -> str:
            return await self.get_response(messages[0]["content"])
    
    # Create test node
    node = GoalTrackerNode(
        id="test_goals",
        agent=TestAgent(),
        memory_key="goals",
        next_nodes={"default": "next_node"}
    )
    
    # Create test state
    state = NodeState()
    state.results["previous"] = {
        "response": "Analysis of GPT-4 capabilities is complete. Beginning impact assessment."
    }
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next_node", "Incorrect next node"
    assert "test_goals" in state.results, "Results not in state"
    assert "goals" in state.context.memory, "Goals not stored in memory"
    print("GoalTrackerNode test passed!")

async def test_pattern_tracker():
    """Test pattern tracking functionality."""
    print("\nTesting PatternTrackerNode...")
    
    # Create test agent
    class TestAgent(BaseAgent):
        async def get_response(self, prompt: str) -> str:
            return """{
                "themes": ["AI capabilities", "Safety considerations"],
                "changes": ["Increased focus on practical applications"],
                "patterns": ["Regular safety discussions", "Technical updates"],
                "deviations": ["Unusual emphasis on deployment speed"]
            }"""
        
        async def _call(self, messages: List[Dict[str, str]], **kwargs) -> str:
            return await self.get_response(messages[0]["content"])
    
    # Create test node
    node = PatternTrackerNode(
        id="test_patterns",
        agent=TestAgent(),
        memory_key="patterns",
        next_nodes={"default": "next_node"}
    )
    
    # Create test state
    state = NodeState()
    state.results["previous"] = {
        "response": "GPT-4 deployment is accelerating, with new capabilities being released rapidly."
    }
    
    # Process node
    next_id = await node.process(state)
    
    # Verify results
    assert next_id == "next_node", "Incorrect next node"
    assert "test_patterns" in state.results, "Results not in state"
    assert "patterns" in state.context.memory, "Patterns not stored in memory"
    print("PatternTrackerNode test passed!")

async def run_all_tests():
    """Run all evaluator node tests."""
    print("Running evaluator node tests...")
    await test_fact_extractor()
    await test_goal_tracker()
    await test_pattern_tracker()
    print("\nAll evaluator node tests passed!")

if __name__ == "__main__":
    print(f"Running tests from: {__file__}")
    import asyncio
    asyncio.run(run_all_tests()) 