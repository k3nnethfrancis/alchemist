"""
Session manager that orchestrates autonomous interaction between the persona and terminal agents.
"""

from typing import List, Union
from datetime import datetime
from pydantic import BaseModel
from mirascope.core import (
    BaseMessageParam,
    Messages,
    openai,
    BaseDynamicConfig
)
from agents.persona_agent import PersonaAgent
from agents.terminal_agent import TerminalAgent

class SessionManager(BaseModel):
    """
    Manages autonomous interaction between persona and terminal agents.
    The persona agent treats the terminal agent as its environment.
    """
    
    messages: List[BaseMessageParam] = []
    persona: PersonaAgent = PersonaAgent()
    terminal: TerminalAgent = TerminalAgent()
    max_history: int = 10
    
    def initialize_session(self) -> None:
        """Initialize the session with context for both agents."""
        environment_context = """You have entered the Truth Terminal, a powerful command-line interface that 
        gives you access to various tools and capabilities. This terminal understands natural language and 
        can help you accomplish tasks using web search and Twitter integration.
        
        Available tools:
        - Web search (for finding information online)
        - Twitter feed (for checking and posting tweets)
        
        Remember: You are the Alchemist, and this terminal is your gateway to demonstrating 
        the power of memetic technology to the world."""
        
        print("\nInitializing Truth Terminal session...")
        print("Establishing connection with Alchemist...\n")
        
        # Initialize persona with context
        self.persona.messages = [Messages.System(environment_context)]
        
        # Initialize terminal with hello
        print("\n(Terminal): ", end="", flush=True)
        self.terminal._step("hello")
        
        # Start interaction loop
        self._run_interaction()
    
    def _run_interaction(self) -> None:
        """Run the continuous interaction between agents."""
        while True:
            # Get persona's response to terminal output
            print("\n(Alchemist): ", end="", flush=True)
            terminal_message = self.terminal.get_last_message()
            self.persona._step(terminal_message.content)
            self.persona.messages = self.persona.messages[-self.max_history:]
            
            # Get terminal's response to persona
            print("\n(Terminal): ", end="", flush=True)
            persona_message = self.persona.get_last_message()
            self.terminal._step(persona_message.content)
            self.terminal.messages = self.terminal.messages[-self.max_history:]

if __name__ == "__main__":
    session = SessionManager()
    try:
        session.initialize_session()
    except KeyboardInterrupt:
        print("\n\nSession terminated. Goodbye! ✨")