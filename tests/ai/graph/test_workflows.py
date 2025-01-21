"""Tests for example graph workflows."""

import asyncio
from datetime import datetime
from typing import Dict, Any, List

from alchemist.ai.graph.base import Graph, NodeState, NodeContext
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.graph.nodes.actions import ActionNode
from alchemist.ai.graph.nodes.decisions import BinaryDecisionNode, MultiChoiceNode
from alchemist.ai.graph.nodes.context import TimeContextNode, MemoryContextNode
from alchemist.ai.graph.nodes.evaluators import EvaluatorNode
from alchemist.ai.base.agent import BaseAgent
from mirascope.core import BaseTool

class TestAgent(BaseAgent):
    """Test agent that returns predefined responses."""
    
    responses: Dict[str, str] = {
        "validate": "yes",
        "analyze": """Key Points:
- GPT-4 represents significant advancement in language understanding
- New capabilities include improved reasoning and domain expertise
- Potential impact on various industries and research fields
- Safety considerations and ethical guidelines highlighted""",
        "categorize": "RESEARCH",
        "extract_facts": """[
{
    "claim": "GPT-4 has improved reasoning capabilities",
    "type": "fact",
    "in_bio": false,
    "already_known": false
},
{
    "claim": "GPT-4 has broader domain expertise",
    "type": "fact", 
    "in_bio": false,
    "already_known": false
},
{
    "claim": "Safety and ethics are key considerations",
    "type": "opinion",
    "in_bio": false,
    "already_known": true
}]""",
        "track_goals": """[{
    "id": "research_analysis",
    "name": "Complete GPT-4 Research Analysis",
    "status": "IN_PROGRESS",
    "objectives": [
        {
            "description": "Gather technical capabilities",
            "completed": true
        },
        {
            "description": "Assess industry impact",
            "completed": true  
        },
        {
            "description": "Review safety considerations",
            "completed": false
        }
    ]
}]""",
        "format": """# GPT-4 Research Summary
## Overview
Latest advancement in AI technology with significant improvements in language understanding and reasoning capabilities.

## Key Findings
1. Enhanced language understanding
2. Improved reasoning capabilities
3. Broader domain expertise

## Impact Analysis
Potential applications across multiple industries with careful consideration of safety and ethics.

## References
- OpenAI Technical Report
- Peer Reviews
- Industry Analysis"""
    }
    
    async def _call(self, messages: list[Dict[str, str]], **kwargs) -> str:
        prompt = messages[0]["content"].lower()
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return "Unexpected prompt"

class TestSearchTool(BaseTool):
    """Simulates research data fetching."""
    
    name: str = "research_search"
    description: str = "Fetches research data and related papers"
    
    async def call(self, query: str, **kwargs) -> Dict[str, Any]:
        return {
            "title": "GPT-4 Technical Report",
            "content": "Detailed analysis of GPT-4 capabilities and benchmarks",
            "sources": ["OpenAI Blog", "arXiv Papers"],
            "timestamp": str(datetime.now())
        }

async def create_research_workflow() -> Graph:
    """Create a research assistant workflow that uses all node types."""
    graph = Graph()
    
    # Context nodes for time and research history
    time_node = TimeContextNode(
        id="time",
        next_nodes={"default": "memory"}
    )
    
    memory_node = MemoryContextNode(
        id="memory",
        memory_keys=["previous_research"],
        next_nodes={"default": "validate"}
    )
    
    # Validate input and fetch data
    validate_node = BinaryDecisionNode(
        id="validate",
        agent=TestAgent(),
        prompt="Is this a valid research topic that needs analysis? Topic: {input}",
        next_nodes={
            "yes": "search",
            "no": "terminal",
            "error": "terminal"
        }
    )
    
    search_node = ActionNode(
        id="search",
        name="research_search",
        tool=TestSearchTool(),
        next_nodes={"default": "analyze"}
    )
    
    # Analysis nodes
    analyze_node = LLMNode(
        id="analyze",
        agent=TestAgent(),
        prompt="Analyze this research data and identify key points: {search_result}",
        next_nodes={"default": "categorize"}
    )
    
    categorize_node = MultiChoiceNode(
        id="categorize",
        agent=TestAgent(),
        prompt="Categorize this research as: RESEARCH, TECHNOLOGY, or NEWS",
        choices=["RESEARCH", "TECHNOLOGY", "NEWS"],
        next_nodes={
            "RESEARCH": "extract_facts",
            "TECHNOLOGY": "extract_facts",
            "NEWS": "terminal"
        }
    )
    
    # Evaluator nodes for fact extraction and goal tracking
    fact_node = EvaluatorNode(
        id="extract_facts",
        agent=TestAgent(),
        prompt="Extract factual claims from this analysis: {analysis}",
        target_key="facts",
        next_nodes={"default": "track_goals"}
    )
    
    goal_node = EvaluatorNode(
        id="track_goals",
        agent=TestAgent(),
        prompt="Track research goals and objectives: {analysis}",
        target_key="goals",
        next_nodes={"default": "format"}
    )
    
    # Final formatting
    format_node = LLMNode(
        id="format",
        agent=TestAgent(),
        prompt="Format this research analysis into a structured report: {analysis}",
        next_nodes={"default": "terminal"}
    )
    
    # Add all nodes
    graph.add_node(time_node)
    graph.add_node(memory_node)
    graph.add_node(validate_node)
    graph.add_node(search_node)
    graph.add_node(analyze_node)
    graph.add_node(categorize_node)
    graph.add_node(fact_node)
    graph.add_node(goal_node)
    graph.add_node(format_node)
    
    return graph

async def test_research_workflow():
    """Test the research assistant workflow."""
    print("\nTesting research assistant workflow...")
    
    # Create initial state with research topic
    state = NodeState(
        context=NodeContext(
            metadata={},
            memory={"previous_research": "Previous GPT-4 analysis from last week"}
        ),
        results={},
        data={"input": "Analyze the capabilities and impact of GPT-4"}
    )
    
    # Create and run workflow
    graph = await create_research_workflow()
    graph.add_entry_point("main", "time")
    
    final_state = await graph.run("main", state)
    
    # Verify workflow execution
    assert "time" in final_state.results, "Time context failed"
    assert "memory" in final_state.results, "Memory context failed"
    assert "validate" in final_state.results, "Validation failed"
    assert "search" in final_state.results, "Search failed"
    assert "analyze" in final_state.results, "Analysis failed"
    assert "categorize" in final_state.results, "Categorization failed"
    assert "extract_facts" in final_state.results, "Fact extraction failed"
    assert "track_goals" in final_state.results, "Goal tracking failed"
    assert "format" in final_state.results, "Formatting failed"
    
    # Verify specific results
    assert final_state.results["validate"]["decision"] == "yes", "Should be valid research topic"
    assert "GPT-4" in final_state.results["analyze"]["response"], "Analysis should mention GPT-4"
    assert final_state.results["categorize"]["choice"] == "RESEARCH", "Should be categorized as RESEARCH"
    
    # Verify evaluator results
    facts = final_state.results["extract_facts"]["facts"]
    assert len(facts) > 0, "Should extract facts"
    assert any(f["type"] == "fact" for f in facts), "Should have factual claims"
    assert any(f["type"] == "opinion" for f in facts), "Should identify opinions"
    
    goals = final_state.results["track_goals"]["goals"]
    assert len(goals) > 0, "Should track goals"
    assert goals[0]["status"] == "IN_PROGRESS", "Research should be in progress"
    assert any(o["completed"] for o in goals[0]["objectives"]), "Should have completed objectives"
    
    print("Research workflow test passed!")

if __name__ == "__main__":
    print(f"Running tests from: {__file__}")
    asyncio.run(test_research_workflow()) 