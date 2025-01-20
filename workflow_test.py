"""Example workflow demonstrating graph system functionality.

This module provides a complete example of how to build and test a content processing
workflow using the Alchemist graph system. It demonstrates:
- Custom LLM nodes for content analysis and filtering
- Tool nodes for content formatting
- Graph construction and execution
- State management between nodes
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directories to path if running directly
if __name__ == "__main__" and __package__ is None:
    file = os.path.abspath(__file__)
    parent = os.path.dirname(os.path.dirname(os.path.dirname(file)))
    sys.path.insert(0, parent)

from alchemist.ai.graph.base import Graph, NodeState
from alchemist.ai.graph.nodes.base.llm import LLMNode
from alchemist.ai.graph.nodes.base.tool import ToolNode
from alchemist.ai.base.agent import BaseAgent

class ContentAnalysisNode(LLMNode):
    """Node for analyzing content using LLM."""
    
    def __init__(self, **data):
        data["prompt"] = "Analyze this content and describe key themes: {content}"
        data["agent"] = BaseAgent()  # Uses default OpenPipe agent
        super().__init__(**data)
        
    async def process(self, state: NodeState) -> Optional[str]:
        logger.info(f"\nAnalyzing content in {self.id}...")
        try:
            next_id = await super().process(state)
            logger.info(f"Analysis complete. Result: {state.results[self.id]}")
            return next_id
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise

class ContentFilterNode(LLMNode):
    """Node for filtering content using LLM."""
    
    def __init__(self, **data):
        data["prompt"] = """Should we include this content? Consider:
1. Is it relevant to tech/AI?
2. Is it informative?
3. Is it appropriate?

Answer only YES or NO: {content}"""
        data["agent"] = BaseAgent()
        super().__init__(**data)
        
    async def process(self, state: NodeState) -> Optional[str]:
        logger.info(f"\nFiltering content in {self.id}...")
        try:
            # Get LLM response using _step
            formatted_prompt = self.prompt.format(**state.results)
            response = await self.agent._step(formatted_prompt)
            
            # Store result
            state.results[self.id] = {"response": response}
            
            # Check response and route accordingly
            response = response.strip().upper()
            next_node = "yes" if response == "YES" else "no"
            logger.info(f"Filter decision: {response}, routing to: {self.next_nodes.get(next_node)}")
            return self.next_nodes.get(next_node)
            
        except Exception as e:
            logger.error(f"Error in filtering: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            return self.next_nodes.get("error")

class ContentCategoryNode(LLMNode):
    """Node for categorizing content using LLM."""
    
    def __init__(self, **data):
        data["prompt"] = """Categorize this content into exactly one category.
Available categories: TECH, NEWS, OTHER

Content: {content}
Analysis: {analyze[response]}

Answer with just the category name."""
        data["agent"] = BaseAgent()
        super().__init__(**data)
        
    async def process(self, state: NodeState) -> Optional[str]:
        logger.info(f"\nCategorizing content in {self.id}...")
        try:
            next_id = await super().process(state)
            if next_id:
                # Route based on category
                response = state.results[self.id]["response"].strip().upper()
                next_node = response.lower()
                logger.info(f"Category decision: {response}, routing to: {next_node}")
                return self.next_nodes.get(next_node, self.next_nodes.get("other"))
            return next_id
        except Exception as e:
            logger.error(f"Error in categorization: {str(e)}")
            raise

class ContentFormattingNode(ToolNode):
    """Node for formatting content using a tool."""
    
    def __init__(self, **data):
        data["tool_name"] = "content_formatter"
        data["tool_args"] = {
            "format": "markdown",
            "content": "{content}",
            "analysis": "{analyze.response}",
            "category": "{categorize.response}"
        }
        super().__init__(**data)

async def build_workflow() -> Graph:
    """Build example content processing workflow."""
    graph = Graph()
    
    # Create nodes
    filter_node = ContentFilterNode(
        id="filter",
        next_nodes={
            "yes": "analyze",
            "no": None,
            "error": None
        }
    )
    
    analyze_node = ContentAnalysisNode(
        id="analyze",
        next_nodes={
            "default": "categorize",
            "error": None
        }
    )
    
    category_node = ContentCategoryNode(
        id="categorize",
        next_nodes={
            "tech": "format",
            "news": "format",
            "other": None,
            "error": None
        }
    )
    
    format_node = ContentFormattingNode(
        id="format",
        next_nodes={
            "default": None,
            "error": None
        }
    )
    
    # Add nodes to graph
    graph.add_node(filter_node)
    graph.add_node(analyze_node)
    graph.add_node(category_node)
    graph.add_node(format_node)
    
    # Add edges
    graph.add_edge("filter", "yes", "analyze")
    graph.add_edge("filter", "no", None)
    graph.add_edge("analyze", "default", "categorize")
    graph.add_edge("categorize", "tech", "format")
    graph.add_edge("categorize", "news", "format")
    graph.add_edge("categorize", "other", None)
    
    # Add entry point
    graph.add_entry_point("main", "filter")
    
    return graph

async def test_workflow():
    """Test the content processing workflow."""
    print("\nTesting Content Processing Workflow...")
    
    # Build workflow
    graph = await build_workflow()
    
    # Test cases
    test_cases = [
        """OpenAI has released GPT-4, their most advanced AI model yet.
        The new model shows significant improvements in reasoning and
        creative tasks while maintaining strong safeguards.""",
        
        """Breaking: Major earthquake hits Pacific region.
        Tsunami warnings issued for coastal areas.
        Emergency services are responding.""",
        
        """The weather is nice today. Birds are chirping.
        I think I'll go for a walk in the park."""
    ]
    
    for i, content in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Input: {content[:50]}...")
        
        # Create state for this test
        state = NodeState()
        state.results["content"] = content
        
        # Execute workflow
        final_state = await graph.run("main", state)
        
        # Print results
        print("\nResults:")
        for node_id, result in final_state.results.items():
            if node_id != "content":
                print(f"\n{node_id.upper()} Node:")
                print(result)
        
        print("\n" + "="*50)
    
    print("\nWorkflow testing completed!")

if __name__ == "__main__":
    print(f"Running workflow test from: {__file__}")
    asyncio.run(test_workflow()) 