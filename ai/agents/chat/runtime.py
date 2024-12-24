"""
Chat Runtime Module

This module handles the runtime execution of the chat agent, including:
- Message processing
- Tool execution
- Response formatting
- History management
"""

from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field

from ai.base.runtime import BaseRuntime
from ai.agents.chat.agent import ChatAgent

import logging
logger = logging.getLogger(__name__)

class ChatRuntime(BaseRuntime):
    """
    Runtime environment for the ChatAgent.
    
    This class handles:
    - Message processing and response generation
    - Tool execution and result handling
    - History management
    - Error handling
    """
    
    def __init__(self, agent: ChatAgent, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chat runtime.
        
        Args:
            agent (ChatAgent): The chat agent instance
            config (Optional[Dict[str, Any]]): Runtime configuration
        """
        super().__init__(agent=agent, config=config)
        self._start_session("cli")  # Default to CLI platform

    async def start(self) -> None:
        """Start the runtime in interactive CLI mode."""
        print("\n" + "="*50)
        print("🌟 Chat Session Started 🌟")
        print("Type 'exit' or 'quit' to end the session")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\033[34mYou: \033[0m")  # Blue for user
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                # Print a newline for logs
                print("\n", end="")
                
                # Process message and get response
                response = await self.process_message(user_input)
                
                # Print a newline after logs
                print("\n", end="")
                
                # Print assistant response in green
                print(f"\033[32mAssistant: {response}\033[0m\n")
                
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                print(f"\n\033[31mError: {str(e)}\033[0m\n")

    async def stop(self) -> None:
        """Stop the runtime."""
        print("\n" + "="*50)
        print("🌟 Chat Session Ended 🌟")
        print("="*50 + "\n")
