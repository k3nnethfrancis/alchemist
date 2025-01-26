"""
Analysis Workflow Example

This demonstrates a crypto price analysis workflow using:
1. ActionNode for fetching mock price data
2. AgentNode for analysis using Mirascope
"""

import asyncio
import random
from datetime import datetime
from pydantic import BaseModel, Field, SkipValidation
from alchemist.ai.graph.state import NodeStatus, NodeState
from alchemist.ai.graph.nodes.actions import ActionNode
from alchemist.ai.graph.nodes.agent import AgentNode
from alchemist.ai.graph.nodes.terminal import TerminalNode
from alchemist.ai.graph.base import Graph
from alchemist.ai.base.agent import BaseAgent
from mirascope.core import prompt_template, Messages
from alchemist.ai.base.logging import (
    configure_logging, 
    LogLevel, 
    LogComponent, 
    get_logger,
    Colors,
    VerbosityLevel,
    AlchemistLoggingConfig
)
from typing import Optional

# Configure logging
configure_logging(
    default_level=LogLevel.INFO,
    component_levels={
        LogComponent.WORKFLOW: LogLevel.INFO,
        LogComponent.TOOLS: LogLevel.INFO,
        LogComponent.NODES: LogLevel.INFO
    }
)

logger = get_logger(LogComponent.WORKFLOW)

class CryptoRequest(BaseModel):
    """Pydantic model for crypto price request."""
    coin_name: str = Field(..., description="Name of the cryptocurrency")
    currency: str = Field(..., description="Fiat currency to convert to")

async def fetch_crypto_price(coin_name: str, currency: str) -> float:
    """Mock crypto price fetch."""
    await asyncio.sleep(0.5)
    price = round(random.uniform(10000, 50000), 2) if coin_name.lower() == "bitcoin" else round(random.uniform(50, 3000), 2)
    return price

@prompt_template()
def analysis_prompt(coin_name: str, currency: str, price: float) -> Messages.Type:
    """Prompt template for crypto analysis."""
    return Messages.User(
        f"Analyze the current price of {coin_name} at {price} {currency}. "
        "Provide a brief market sentiment analysis."
    )

def log_analysis_step(step_name: str):
    """Create a callback for logging analysis steps."""
    async def callback(state: NodeState, node_id: str) -> None:
        if node_id in state.results:
            result = state.results[node_id]
            print(f"\n{Colors.BOLD}{'=' * 50}{Colors.RESET}")
            print(f"{Colors.BOLD}📊 {step_name}{Colors.RESET}")
            print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}")
            
            if 'output' in result:  # For price data
                print(f"{Colors.INFO}Price: ${result['output']:,.2f} USD{Colors.RESET}")
            if 'response' in result:  # For analysis
                print(f"{Colors.INFO}{result['response']}{Colors.RESET}")
            
            if 'timing' in result:
                print(f"{Colors.DIM}Time: {result['timing']:.1f}s{Colors.RESET}")
            print(f"{Colors.BOLD}{'=' * 50}{Colors.RESET}\n")
            
            logger.info(f"\n📊 {step_name} completed in {result.get('timing', 0):.1f}s")
    return callback

class CryptoPriceNode(ActionNode):
    """Custom ActionNode for fetching crypto prices."""
    
    tool: SkipValidation[callable]  # Wrap callable with SkipValidation
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the crypto price request."""
        try:
            start_time = datetime.now()
            input_data = self._prepare_input_data(state)
            price = await self.tool(**input_data)
            
            state.results[self.id] = {
                "output": price,
                "timing": (datetime.now() - start_time).total_seconds()
            }
            state.mark_status(self.id, NodeStatus.COMPLETED)
            return self.get_next_node()
        except Exception as e:
            state.errors[self.id] = str(e)
            state.mark_status(self.id, NodeStatus.ERROR)
            return self.get_next_node("error")

async def run_analysis_workflow():
    """Run the crypto analysis workflow."""
    
    graph = Graph(
        logging_config=AlchemistLoggingConfig(
            level=VerbosityLevel.INFO,
            show_llm_messages=True,
            show_node_transitions=True,
            show_tool_calls=True
        )
    )
    
    agent = BaseAgent()
    
    # Define nodes
    fetch_node = CryptoPriceNode(
        id="fetch_crypto",
        tool=fetch_crypto_price,
        input_map={
            "coin_name": "data.request.coin_name",
            "currency": "data.request.currency"
        },
        metadata={"on_complete": log_analysis_step("Price Fetch")}
    )

    analyze_node = AgentNode(
        id="analyze",
        prompt=f"Analyze the current price of {{coin_name}} at {{price}} {{currency}}. Provide a brief market sentiment analysis.",
        agent=agent,
        input_map={
            "coin_name": "data.request.coin_name",
            "currency": "data.request.currency",
            "price": "node.fetch_crypto.output"
        },
        metadata={"on_complete": log_analysis_step("Market Analysis")}
    )

    end_node = TerminalNode(id="end")

    # Set up node transitions
    fetch_node.next_nodes = {"default": "analyze", "error": "end"}
    analyze_node.next_nodes = {"default": "end", "error": "end"}

    # Add nodes to graph
    for node in [fetch_node, analyze_node, end_node]:
        graph.add_node(node)
    graph.add_entry_point("start", "fetch_crypto")

    # Create and initialize state
    state = NodeState()
    state.set_data("request", {
        "coin_name": "Bitcoin",
        "currency": "USD"
    })

    print(f"\n{Colors.BOLD}🔍 Analyzing Crypto:{Colors.RESET}")
    print(f"{Colors.INFO}Coin: {state.data['request']['coin_name']}")
    print(f"Currency: {state.data['request']['currency']}{Colors.RESET}\n")
    
    start_time = datetime.now()
    final_state = await graph.run("start", state)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{Colors.SUCCESS}✨ Analysis Complete in {elapsed:.1f}s{Colors.RESET}\n")

if __name__ == "__main__":
    asyncio.run(run_analysis_workflow()) 