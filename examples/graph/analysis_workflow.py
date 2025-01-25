"""
Analysis Workflow Example

References:
    - @dev.txt: Describes overall Alchemist project structure, LLM and tool abstractions
    - @graph_build_notes.txt: Explains the Graph system design, node composition, and state management

Overview:
    This file demonstrates a simple "analysis" workflow using Alchemist's Graph and Node system.  
    It uses:
        1) ToolNode to fetch mock cryptocurrency data (e.g., from an external API).  
        2) LLMNode to perform analysis on that data using Mirascope (provider-agnostic).  
        3) A final ToolNode (or ActionNode) to do some post-processing or summary step.  

    All nodes use Pydantic-based validation and asynchronous processing.  
    The workflow is provider-agnostic, thanks to Mirascope, and can be configured to run 
    with any supported LLM model (gpt-4o-mini, claude-3-5-sonnet-20240620, openpipe, etc.).  

Usage:
    - Run this as a standalone script:  
        >>> python analysis_workflow.py  
    - Or import and call `asyncio.run(run_analysis_workflow())` in your own code.
"""

import asyncio
import random
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from alchemist.ai.graph.base import Graph
from alchemist.ai.graph.state import NodeStatus, NodeState
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.nodes.tool import ToolNode
from alchemist.ai.graph.nodes.llm import LLMNode
from alchemist.ai.graph.nodes.tool import ToolNode as BaseToolNode
from alchemist.ai.base.agent import BaseAgent
from mirascope.core import prompt_template, Messages


class CryptoRequest(BaseModel):
    """
    Pydantic model for requesting a crypto price.

    Attributes:
        coin_name: The name of the cryptocurrency, e.g., "Bitcoin".
        currency: The fiat currency for the price, e.g., "USD".
    """
    coin_name: str = Field(..., description="Name of the cryptocurrency.")
    currency: str = Field(..., description="Fiat currency to convert to, e.g. USD or EUR.")


async def fetch_crypto_price(coin_name: str, currency: str) -> float:
    """
    Mock async function to simulate fetching the current price of a cryptocurrency.
    In a real implementation, this might call an external API such as CoinGecko or CoinMarketCap.

    Args:
        coin_name: The name of the cryptocurrency.
        currency: The fiat currency to convert to.

    Returns:
        A float representing the current price.
    """
    # Simulate I/O
    await asyncio.sleep(0.5)
    # Return a pseudo-random price for demonstration:
    return round(random.uniform(10000, 50000), 2) if coin_name.lower() == "bitcoin" else round(random.uniform(50, 3000), 2)


class CryptoPriceToolNode(BaseToolNode):
    """
    A ToolNode specialized for fetching a mock crypto price.

    Attributes:
        request_key: The key in NodeState where the CryptoRequest data is stored.
        output_key: Where to store the result in state.results under this node's ID.
    """
    request_key: str = "crypto_request"
    output_key: str = "price"

    async def process(self, state: NodeState) -> Optional[str]:
        """Process the crypto price request."""
        try:
            # Get input data using base Node's helper
            input_data = self._prepare_input_data(state)
            
            # Pydantic validation
            req = CryptoRequest(**input_data)
            
            # Fetch price using the tool
            price = await fetch_crypto_price(req.coin_name, req.currency)
            
            # Store result and mark status
            state.results[self.id] = {self.output_key: price}
            state.mark_status(self.id, NodeStatus.COMPLETED)
            
            return self.get_next_node()

        except Exception as e:
            state.errors[self.id] = str(e)
            state.mark_status(self.id, NodeStatus.ERROR)
            return self.get_next_node("error")


@prompt_template()
def analysis_prompt(coin_name: str, currency: str, price: float) -> Messages.Type:
    """
    A Mirascope prompt template that instructs the LLM to analyze the fetched crypto price.

    Args:
        coin_name: Cryptocurrency name, e.g., "Bitcoin".
        currency: The chosen fiat currency, e.g., "USD".
        price: The fetched crypto price.

    Returns:
        A Mirascope Messages object that the agent can process.
    """
    return Messages.User(
        f"Analyze the {coin_name} market. The current price is {price} {currency}. "
        "Provide a short summary or insight about this price."
    )


class AnalysisNode(LLMNode):
    """
    LLMNode that uses a Mirascope prompt_template to analyze the fetched crypto price.
    """
    async def process(self, state: NodeState) -> Optional[str]:
        try:
            # Get the crypto request data
            request_data = state.get_data("crypto_request")
            if not request_data:
                raise ValueError("No crypto request data found")

            # Get the price from the fetch node's results
            price = state.get_result("fetch_crypto", "price")
            if price is None:
                raise ValueError("No price data found from fetch_crypto node")

            # Prepare the template data
            template_data = {
                "coin_name": request_data["coin_name"],
                "currency": request_data["currency"],
                "price": price
            }

            # Use parent class process with our prepared data
            state.set_data("template_data", template_data)
            return await super().process(state)

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            state.add_error(self.id, str(e))
            return self.get_next_node("error")


class TerminalNode(Node):
    """
    Terminal node that ends the workflow with no further transitions.
    """
    async def process(self, state: NodeState) -> Optional[str]:
        # Mark this node as terminal
        state.mark_status(self.id, NodeStatus.TERMINAL)
        return None


async def run_analysis_workflow() -> None:
    """
    Main entrypoint for demonstrating a crypto analysis workflow using ToolNode + LLMNode
    in the Alchemist Graph system.

    Steps:
        1. The user or host code sets up NodeState with a CryptoRequest.
        2. A ToolNode (CryptoPriceToolNode) fetches the price and stores it in NodeState.
        3. An LLMNode (AnalysisNode) uses Mirascope to analyze that price.
        4. The final NodeState is returned and printed to the console.

    Raises:
        AssertionError: If the Graph validation fails or an unexpected error occurs.
    """
    # 1. Create the Graph
    graph = Graph()

    # 2. Define the nodes
    fetch_node = CryptoPriceToolNode(
        id="fetch_crypto",
        tool=fetch_crypto_price,
        input_map={
            "coin_name": "crypto_request.coin_name",
            "currency": "crypto_request.currency"
        },
        next_nodes={"default": "analyze", "error": "end"}
    )

    analyze_node = AnalysisNode(
        id="analyze",
        prompt_template=analysis_prompt,
        system_prompt="You are a helpful financial assistant.",
        input_map={
            "coin_name": "crypto_request.coin_name",
            "currency": "crypto_request.currency",
            "price": "fetch_crypto.price"
        },
        next_nodes={"default": "end", "error": "end"}
    )

    #    - Terminal node
    end_node = TerminalNode(
        id="end",
        next_nodes={}
    )

    # 3. Add the nodes to the graph
    graph.add_node(fetch_node)
    graph.add_node(analyze_node)
    graph.add_node(end_node)

    # 4. Add an entry point
    graph.add_entry_point("start", "fetch_crypto")

    # 5. Validate the graph
    errors = graph.validate()
    if errors:
        raise AssertionError(f"Graph validation failed: {errors}")

    # 6. Prepare the NodeState
    state = NodeState()
    state.set_data(
        "crypto_request",
        {
            "coin_name": "Bitcoin",  # try adjusting to "Ethereum" or others
            "currency": "USD"
        },
    )

    # 7. Run the graph
    final_state = await graph.run("start", state)

    # 8. Print results
    print("Analysis Workflow Completed.\n")
    print("Node States and Results:")
    for node_id, result_data in final_state.results.items():
        print(f"- {node_id}: {result_data}")
    if final_state.errors:
        print("\nErrors encountered:")
        for node_id, err_msg in final_state.errors.items():
            print(f"  {node_id}: {err_msg}")


if __name__ == "__main__":
    """
    Execute the analysis workflow as a standalone script:

    This demonstrates how to build a reproducible example of a multi-step AI workflow,
    including a tool-based data fetch and an LLM-based analysis with Mirascope.
    """
    asyncio.run(run_analysis_workflow()) 