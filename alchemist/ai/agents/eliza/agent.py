"""Eliza agent implementation."""

import logging
from typing import Optional, Dict, Any

from pydantic import Field
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.agents.eliza.workflow import create_eliza_workflow
from alchemist.ai.graph.base import Graph, NodeContext

logger = logging.getLogger(__name__)

class ElizaAgent(BaseAgent):
    """Modern implementation of the ELIZA chatbot using LLMs and graph-based workflow."""
    
    workflow: Graph = Field(default_factory=create_eliza_workflow)
    messages: list = Field(default_factory=list)
    last_check_time: int = Field(default=0)
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", **kwargs):
        """Initialize the Eliza agent.
        
        Args:
            provider: The LLM provider to use
            model: The model to use
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(provider=provider, model=model, **kwargs)
        logger.debug(f"Initialized ElizaAgent with provider={provider}, model={model}")
        
    async def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the conversation history.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        logger.debug(f"Added message to history: {message}")
        
    async def process_discord_message(self, message: Dict[str, Any]) -> Optional[str]:
        """Process a Discord message and generate a response if appropriate.
        
        Args:
            message: The Discord message to process
            
        Returns:
            Optional response string
        """
        try:
            # Add message to history
            await self.add_message(message)
            
            # Always respond if mentioned
            if message.get("is_mention", False):
                logger.info("Bot was mentioned, generating response")
                
                # Create context with metadata
                context = NodeContext(
                    metadata={
                        "current_message": message.get("content", ""),
                        "is_mention": True,
                        "messages": self.messages
                    }
                )
                logger.debug(f"Running workflow with context: {context}")
                
                state = await self.workflow.run("main", context=context)
                logger.debug(f"Workflow results: {state.results if state else None}")
                
                # Get response from generate node
                if state and "generate_response" in state.results:
                    response = state.results["generate_response"].get("response")
                    logger.info(f"Generated response for mention: {response}")
                    return response
                    
            # Otherwise check cooldown
            current_time = message.get("timestamp", 0)
            if current_time - self.last_check_time < 60:  # 60 second cooldown
                logger.debug("Message within cooldown period, skipping")
                return None
                
            self.last_check_time = current_time
            
            # Create context with metadata
            context = NodeContext(
                metadata={
                    "current_message": message.get("content", ""),
                    "is_mention": False,
                    "messages": self.messages
                }
            )
            logger.debug(f"Running workflow with context: {context}")
            
            state = await self.workflow.run("main", context=context)
            logger.debug(f"Workflow results: {state.results if state else None}")
            
            # Get response if workflow decided to generate one
            if state and "generate_response" in state.results:
                response = state.results["generate_response"].get("response")
                logger.info(f"Generated response for non-mention: {response}")
                return response
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return None 