"""
Analysis Workflow Example

This demonstrates a simple crypto price analysis workflow using:
1. ToolNode for fetching mock price data
2. LLMNode for analysis using Mirascope
"""

import asyncio
import random
from pydantic import BaseModel, Field
from alchemist.ai.graph.state import NodeStatus, NodeState
from alchemist.ai.graph.nodes.tool import ToolNode
from alchemist.ai.graph.nodes.llm import LLMNode
from alchemist.ai.graph.nodes.terminal import TerminalNode
from mirascope.core import prompt_template, Messages
from alchemist.ai.base.logging import get_logger, LogComponent, configure_logging, LogLevel
from typing import Optional

# Set up detailed logging
configure_logging(
    default_level=LogLevel.DEBUG,
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set up logger
logger = get_logger(LogComponent.WORKFLOW)

class CryptoRequest(BaseModel):
    """Pydantic model for crypto price request."""
    coin_name: str = Field(..., description="Name of the cryptocurrency.")
    currency: str = Field(..., description="Fiat currency to convert to.")

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

class CryptoPriceNode(ToolNode):
    """Custom ToolNode for fetching crypto prices."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the crypto price request."""
        try:
            input_data = self._prepare_input_data(state)
            price = await fetch_crypto_price(**input_data)
            
            # Store with standard output structure
            state.results[self.id] = {
                "output": price  # Use consistent 'output' key
            }
            state.mark_status(self.id, NodeStatus.COMPLETED)
            return self.get_next_node()
        except Exception as e:
            state.errors[self.id] = str(e)
            state.mark_status(self.id, NodeStatus.ERROR)
            return self.get_next_node("error")

async def run_analysis_workflow():
    """Run the crypto analysis workflow."""
    
    # Define nodes
    fetch_node = CryptoPriceNode(
        id="fetch_crypto",
        tool=fetch_crypto_price,
        input_map={
            "coin_name": "request.coin_name",
            "currency": "request.currency"
        }
    )

    analyze_node = LLMNode(
        id="analyze",
        prompt_template=analysis_prompt,
        system_prompt="You are a cryptocurrency market analyst.",
        input_map={
            "coin_name": "request.coin_name",
            "currency": "request.currency",
            "price": "{fetch_crypto}.output"  # Use curly braces to indicate node result lookup
        }
    )

    end_node = TerminalNode(id="end")

    # Set up node transitions
    fetch_node.next_nodes = {"default": "analyze", "error": "end"}
    analyze_node.next_nodes = {"default": "end", "error": "end"}

    # Create and initialize state
    state = NodeState()
    state.set_data("request", {
        "coin_name": "Bitcoin",
        "currency": "USD"
    })

    # Process nodes in sequence
    current_node = fetch_node
    while current_node:
        # Log state before each node
        logger.debug(f"Before {current_node.id} - State data: {state.data}")
        logger.debug(f"Before {current_node.id} - State results: {state.results}")
        
        next_id = await current_node.process(state)
        
        # Log state after each node
        logger.debug(f"After {current_node.id} - State results: {state.results}")
        logger.debug(f"After {current_node.id} - Next node: {next_id}")
        
        if next_id == "analyze":
            current_node = analyze_node
        elif next_id == "end":
            current_node = end_node
        else:
            break

    # Print results
    print("\nWorkflow Results:")
    if 'fetch_crypto' in state.results:
        print(f"Crypto Price: {state.results['fetch_crypto']['output']} USD")
    if 'analyze' in state.results and 'response' in state.results['analyze']:
        print(f"Analysis: {state.results['analyze']['response']}")
    else:
        print("Analysis: Failed to generate analysis")
        print(f"State results: {state.results}")  # Debug info

if __name__ == "__main__":
    asyncio.run(run_analysis_workflow()) 