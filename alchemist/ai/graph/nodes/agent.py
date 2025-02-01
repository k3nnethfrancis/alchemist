"""
Agent Node Implementation

This module provides the AgentNode class for executing LLM agent steps within a graph.
Supports:
- Prompt templating with input mapping
- Streaming responses
- Structured outputs
- Tool execution
"""

import logging
from typing import Optional, Dict, Any, Union, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.logging import get_logger, LogComponent, Colors
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.state import NodeState, NodeStatus

logger = get_logger(LogComponent.NODES)

class AgentNode(Node):
    """
    Node for executing LLM agent steps with support for streaming and structured outputs.
    
    Features:
        - Prompt templating with state mapping
        - Streaming response support
        - Structured output validation
        - Tool execution handling
        - Response timing and logging
    """
    
    prompt: str = Field(..., description="Prompt template to send to the agent")
    agent: BaseAgent = Field(..., description="Agent instance to use for LLM calls")
    stream: bool = Field(default=False, description="Whether to stream the response")
    response_model: Optional[type[BaseModel]] = Field(
        default=None, 
        description="Optional Pydantic model for structured output"
    )
    json_mode: bool = Field(
        default=False,
        description="Whether to enforce JSON output format"
    )

    async def process(self, state: NodeState) -> Optional[str]:
        """
        Process the node by executing an agent step.
        
        Args:
            state: Current graph state
            
        Returns:
            Optional[str]: Next node ID or None
            
        Features:
            - Supports streaming responses
            - Handles structured outputs
            - Tracks timing and status
            - Emits progress events
        """
        try:
            start_time = datetime.now()
            state.mark_status(self.id, NodeStatus.RUNNING)
            
            # Format prompt with input mapping
            inputs = self._prepare_input_data(state)
            formatted_prompt = self.prompt.format(**inputs)
            
            # Configure agent for this step if needed
            if self.response_model:
                self.agent.response_model = self.response_model
            if self.json_mode:
                self.agent.json_mode = self.json_mode
            
            # Log start of processing
            logger.debug(
                f"\n{Colors.BOLD}🤖 Node {self.id} Starting:{Colors.RESET}\n"
                f"{Colors.DIM}Prompt: {formatted_prompt}{Colors.RESET}"
            )
            
            # Process with streaming or standard mode
            if self.stream:
                chunks = []
                async for chunk, tool in self.agent._stream_step(formatted_prompt):
                    if tool:
                        logger.info(f"[Calling Tool '{tool._name()}' with args {tool.args}]")
                        # Handle tool execution
                        if hasattr(tool, 'call'):
                            result = await tool.call() if asyncio.iscoroutinefunction(tool.call) else tool.call()
                            logger.info(f"Tool result: {result}")
                    else:
                        chunks.append(chunk)
                        # Emit streaming event
                        if "on_stream" in self.metadata:
                            await self.metadata["on_stream"](chunk, state, self.id)
                
                # Store complete response
                response = "".join(chunks)
            else:
                # Standard non-streaming call
                response = await self.agent._step(formatted_prompt)
            
            # Store results with timing
            elapsed = (datetime.now() - start_time).total_seconds()
            state.results[self.id] = {
                "response": response,
                "timing": elapsed
            }
            
            # Log completion
            logger.info(
                f"\n{Colors.SUCCESS}✓ Node '{self.id}' completed in {elapsed:.2f}s{Colors.RESET}"
                f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}"
                f"\n{Colors.INFO}{response}{Colors.RESET}"
                f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}\n"
            )
            
            # Handle completion callback
            if "on_complete" in self.metadata:
                await self.metadata["on_complete"](state, self.id)
            
            state.mark_status(self.id, NodeStatus.COMPLETED)
            return self.get_next_node()
            
        except Exception as e:
            logger.error(f"Error in agent node '{self.id}': {str(e)}")
            state.errors[self.id] = str(e)
            state.mark_status(self.id, NodeStatus.ERROR)
            return self.get_next_node("error")

async def test_agent_node():
    """Test the AgentNode with streaming and structured output."""
    from alchemist.ai.base.agent import BaseAgent
    from alchemist.ai.graph.state import NodeState
    from pydantic import BaseModel
    
    # Define a test response model
    class TestResponse(BaseModel):
        message: str
        confidence: float
    
    # Create test agent and node
    agent = BaseAgent()
    node = AgentNode(
        id="test_agent",
        prompt="Respond to: {message}",
        agent=agent,
        stream=True,
        response_model=TestResponse,
        input_map={"message": "data.user_message"},
        next_nodes={"default": "next", "error": "error"}
    )
    
    # Create test state
    state = NodeState()
    state.data["user_message"] = "Tell me about AI."
    
    # Run test
    next_id = await node.process(state)
    print(f"Next node: {next_id}")
    print(f"Results: {state.results}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_node()) 