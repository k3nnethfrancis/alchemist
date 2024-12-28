"""Tests for example graph workflows."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from alchemist.ai.graph.base import Graph, NodeState, NodeContext
from alchemist.ai.base.agent import BaseAgent

@pytest.fixture
def test_agent():
    """Create a test agent."""
    agent = MagicMock(spec=BaseAgent)
    agent.get_response = AsyncMock(return_value="Test response")
    agent._call_tool = AsyncMock(return_value={"data": "test data"})
    agent._call_llm = AsyncMock(return_value="yes")
    return agent

class TestAnalysisWorkflow:
    """Test suite for crypto analysis workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, test_agent):
        """Test complete workflow execution."""
        from examples.graph.analysis_workflow import create_analysis_workflow
        
        # Create initial state with required metadata
        state = NodeState(
            context=NodeContext(
                metadata={
                    "summarize": {
                        "response": "Market looks bullish",
                        "market_trend": "bullish",
                        "key_levels": "50000",
                        "risk_factors": "low"
                    }
                }
            ),
            results={},
            data={}
        )
        
        # Mock responses for each step
        test_agent._call_tool.return_value = {
            "data": {
                "price": 50000,
                "volume": 1000000,
                "timestamp": str(datetime.now())
            }
        }
        
        responses = {
            "analyze": "Price is trending up with strong volume",
            "summarize": "Market looks bullish with strong momentum",
            "decide": "yes"  # Trading decision
        }
        
        def mock_response(prompt: str) -> str:
            if "analyze this crypto" in prompt.lower():
                return responses["analyze"]
            elif "provide a brief market summary" in prompt.lower():
                return responses["summarize"]
            elif "trading position" in prompt.lower():
                return responses["decide"]
            return "Unexpected prompt"
            
        test_agent.get_response = AsyncMock(side_effect=mock_response)
        
        # Create and run workflow
        graph = await create_analysis_workflow()
        
        # Add entry point
        graph.add_entry_point("main", "fetch_data")
        
        final_state = await graph.run("main", state)
        
        # Verify workflow execution
        assert "fetch_data" in final_state.results
        assert "analyze" in final_state.results
        assert "summarize" in final_state.results
        assert "decide" in final_state.results
        
        # Verify decision
        assert final_state.results["decide"]["decision"] == "yes" 