"""System-wide logging test script.

This script demonstrates and tests logging across all major components of the Alchemist system.
It creates a simple workflow that exercises each component while showing log output.
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from alchemist.ai.base.logging import (
    configure_logging,
    LogComponent,
    LogLevel,
    LogFormat,
    get_logger
)
from alchemist.ai.base.runtime import RuntimeConfig
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import Graph, NodeState, NodeContext, Node
from alchemist.ai.tools.calculator import CalculatorTool
from alchemist.ai.prompts.persona import ALCHEMIST

class TestNode(Node):
    """A test node for logging tests."""
    
    def __init__(self, id: str):
        """Initialize the test node."""
        super().__init__(id=id, next_nodes={})
        self._logger = logging.getLogger(__name__)

    async def process(self, state: NodeState) -> Optional[str]:
        """Process the test node."""
        try:
            self._logger.info(f"Processing test node: {self.id}")
            timestamp = datetime.now().isoformat()
            state.results[self.id] = {
                "node": self.id,
                "timestamp": timestamp,
                "status": "success"
            }
            self._logger.debug(f"Test node result: {state.results[self.id]}")
            return self.next_nodes.get("default")
        except Exception as e:
            self._logger.error(f"Error in test node {self.id}: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            return self.next_nodes.get("error")

async def test_agent_logging(log_dir: Path):
    """Test logging in the agent component."""
    logger = get_logger(LogComponent.AGENT)
    logger.info("Testing agent logging...")
    
    # Configure agent
    config = RuntimeConfig(
        provider="openpipe",
        model="gpt-4o-mini",
        persona=ALCHEMIST,
        tools=[CalculatorTool]
    )
    
    agent = BaseAgent(runtime_config=config.model_dump())
    logger.debug(f"Agent initialized with config: {config.model_dump()}")
    
    # Test agent interaction
    response = await agent._step("What is 2 + 2?")
    logger.info(f"Agent response: {response}")

async def test_graph_logging(log_dir: Path):
    """Test logging in the graph system."""
    logger = get_logger(LogComponent.GRAPH)
    logger.info("Testing graph logging...")
    
    # Create a simple test graph
    graph = Graph()
    logger.debug("Created new graph")
    
    # Add a test node
    test_node = TestNode(id="test")
    graph.add_node(test_node)
    logger.debug(f"Added node: {test_node.id}")
    
    # Add entry point
    graph.add_entry_point("main", "test")
    logger.debug("Added entry point")
    
    # Create and run state
    state = NodeState(context=NodeContext())
    logger.debug("Created initial state")
    
    final_state = await graph.run("main", state)
    logger.info("Graph execution complete")

async def test_tool_logging(log_dir: Path):
    """Test logging in the tools system."""
    logger = get_logger(LogComponent.TOOLS)
    logger.info("Testing tool logging...")
    
    # Create and use calculator tool with default expression
    calc = CalculatorTool(expression="1 + 1")
    logger.debug("Calculator tool initialized")
    
    # Update expression and calculate
    calc.expression = "2 + 2"
    result = calc.call()
    logger.info(f"Calculator result: {result}")

async def test_workflow_logging(log_dir: Path):
    """Test logging in the workflow system."""
    logger = get_logger(LogComponent.WORKFLOW)
    logger.info("Testing workflow logging...")
    
    # Create a simple workflow
    workflow = Graph()
    logger.debug("Created workflow graph")
    
    # Add some test nodes
    node1 = TestNode(id="step1")
    node2 = TestNode(id="step2")
    
    workflow.add_node(node1)
    workflow.add_node(node2)
    logger.debug("Added workflow nodes")
    
    # Add entry point
    workflow.add_entry_point("main", "step1")
    logger.debug("Added workflow entry point")
    
    # Run workflow
    state = NodeState(context=NodeContext())
    await workflow.run("main", state)
    logger.info("Workflow execution complete")

async def main():
    """Run all logging tests."""
    # Create log directory
    log_dir = Path("logs/system_test")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging with all outputs
    configure_logging(
        default_level=LogLevel.DEBUG,
        component_levels={
            LogComponent.AGENT: LogLevel.DEBUG,
            LogComponent.GRAPH: LogLevel.DEBUG,
            LogComponent.TOOLS: LogLevel.DEBUG,
            LogComponent.WORKFLOW: LogLevel.DEBUG
        },
        format_string=LogFormat.DEBUG,
        log_file=str(log_dir / f"system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        enable_json=True
    )
    
    # Run component tests
    print("\nTesting Agent Logging...")
    await test_agent_logging(log_dir)
    
    print("\nTesting Graph Logging...")
    await test_graph_logging(log_dir)
    
    print("\nTesting Tool Logging...")
    await test_tool_logging(log_dir)
    
    print("\nTesting Workflow Logging...")
    await test_workflow_logging(log_dir)
    
    print(f"\nLog files can be found in: {log_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 