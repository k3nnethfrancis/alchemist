"""Example of a crypto trading analysis workflow using graph nodes."""

from typing import Dict, Any
from alchemist.ai.graph.base import Graph, NodeState, NodeContext
import logging

from alchemist.ai.graph.nodes.actions import ToolNode
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.graph.nodes.decisions import BinaryDecisionNode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_analysis_workflow() -> Graph:
    """Create a crypto trading analysis workflow."""
    logger.info("Creating crypto analysis workflow...")
    
    # Create nodes
    fetch_data = ToolNode(
        id="fetch_data",
        tool_name="crypto.fetch",
        tool_args={"symbol": "BTC/USD", "timeframe": "1h"},
        next_nodes={"default": "analyze"}
    )
    
    analyze = LLMNode(
        id="analyze",
        prompt="""Analyze this crypto market data and provide key insights:
        Price Data: {fetch_data[data]}
        
        Focus on:
        1. Price trends
        2. Volume patterns
        3. Key support/resistance levels""",
        next_nodes={"default": "summarize"}
    )
    
    summarize = LLMNode(
        id="summarize",
        prompt="""Based on these insights, provide a concise trading summary:
        Analysis: {analyze[response]}
        
        Format as:
        - Market Trend:
        - Key Levels:
        - Risk Factors:""",
        next_nodes={"default": "decide"}
    )
    
    decide = BinaryDecisionNode(
        id="decide",
        prompt="""Based on the analysis, should we take a trading position?
        Summary: {summarize[response]}
        
        Consider:
        1. Clear trend direction
        2. Risk/reward ratio
        3. Market conditions
        
        Respond 'yes' to enter a trade or 'no' to wait.""",
        next_nodes={
            "yes": None,  # End workflow with trade signal
            "no": None,   # End workflow with hold signal
            "error": None
        }
    )
    
    # Create and validate graph
    graph = Graph()
    graph.add_node(fetch_data)
    graph.add_node(analyze)
    graph.add_node(summarize)
    graph.add_node(decide)
    
    # Add entry point
    graph.add_entry_point("main", "fetch_data")
    
    logger.info("Validating graph...")
    graph.validate()
    
    return graph

async def run_workflow():
    """Run the crypto analysis workflow."""
    logger.info("Starting workflow execution...")
    
    graph = await create_analysis_workflow()
    
    # Initial state
    state = NodeState(
        context=NodeContext(),
        results={},
        data={}
    )
    
    # Single run of the analysis
    logger.info("\nExecuting analysis workflow...")
    final_state = await graph.run("main", state)
    
    # Print results
    logger.info("\nWorkflow Results:")
    for node_id, result in final_state.results.items():
        logger.info(f"\n{node_id}:")
        logger.info(result)
    
    # Check final decision
    decision = final_state.results.get("decide", {}).get("decision")
    if decision == "yes":
        logger.info("\nFinal Decision: ENTER TRADE 📈")
    else:
        logger.info("\nFinal Decision: HOLD POSITION 🔄")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_workflow()) 