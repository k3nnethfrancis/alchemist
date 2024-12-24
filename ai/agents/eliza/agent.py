'''
ElizaAgent class for handling Discord interactions.

This module implements the core Eliza agent functionality including:
- Decision making for when to engage in conversation
- Message processing and response generation
- Cooldown management
- Proactive behavior loop
'''

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from ai.prompts.persona import Persona


class ElizaConfig(BaseModel):
    """Configuration for the ElizaAgent."""
    provider: str = Field(default="openpipe")
    persona: Union[Persona, List[str]] = Field(default_factory=list)
    cooldown_seconds: int = Field(default=3)
    scan_interval_seconds: int = Field(default=120)
    history: List[Dict] = Field(default_factory=list)

    def get_persona_traits(self) -> List[str]:
        """Get persona traits as a list of strings."""
        if isinstance(self.persona, Persona):
            return (
                self.persona.style.all +
                self.persona.style.chat +
                [f"You are {self.persona.name} ({self.persona.nickname}), {self.persona.bio}"] +
                [f"Background: {lore}" for lore in self.persona.lore]
            )
        return self.persona


class ElizaAgent(BaseModel):
    """
    An AI agent that can engage in conversation on Discord.
    
    Attributes:
        config: Configuration settings
        last_response: Timestamp of last response
        cooldown: Minimum time between responses
        scan_interval: Time between proactive scans
        history: List of conversation messages
    """
    config: ElizaConfig = Field(default_factory=ElizaConfig)
    last_response: Optional[datetime] = Field(default=None)
    cooldown: timedelta = Field(default_factory=lambda: timedelta(seconds=3))
    scan_interval: timedelta = Field(default_factory=lambda: timedelta(seconds=120))
    history: List[Dict] = Field(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(
        self,
        provider: str = "openpipe",
        persona: Optional[List[str]] = None,
        cooldown_seconds: int = 3,
        scan_interval_seconds: int = 120,
        history: Optional[List[Dict]] = None
    ):
        """
        Initialize the ElizaAgent.
    
        Args:
            provider: The LLM provider to use
            persona: List of personality traits
            cooldown_seconds: Minimum time between responses
            scan_interval_seconds: Time between proactive scans
            history: Optional list of previous messages
        """
        super().__init__(
            config=ElizaConfig(
                provider=provider,
                persona=persona or [],
                cooldown_seconds=cooldown_seconds,
                scan_interval_seconds=scan_interval_seconds,
                history=history or []
            ),
            cooldown=timedelta(seconds=cooldown_seconds),
            scan_interval=timedelta(seconds=scan_interval_seconds),
            history=history or []
        )

    async def _call_provider(self, messages: List[Dict], system_prompt: Optional[str] = None) -> Dict:
        """
        Call the LLM provider with the given messages.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt to use
            
        Returns:
            Provider response
        """
        try:
            # Add system prompt if provided
            if system_prompt:
                messages = [{
                    "role": "system",
                    "content": system_prompt
                }] + messages
            
            # Add persona to system prompt
            if self.config.persona:
                persona_prompt = "\\n".join([
                    "You are an AI assistant with the following traits:",
                    *[f"- {trait}" for trait in self.config.persona]
                ])
                
                messages = [{
                    "role": "system",
                    "content": persona_prompt
                }] + messages
            
            # Mock response for testing
            # TODO: Replace with actual provider call
            if messages[-1]["role"] == "system":
                # Decision prompt
                return {
                    "should_respond": True,
                    "reasoning": "Natural conversation opening",
                    "response_type": "direct"
                }
            else:
                # Response prompt
                return {
                    "content": "Hello! How can I help?"
                }
            
        except Exception as e:
            print(f"Error calling provider: {str(e)}")
            return {
                "error": str(e)
            }

    async def should_respond(self, messages: List[Dict], channel_context: str) -> Dict:
        """
        Determine if the agent should respond to the given messages.
        
        Args:
            messages: List of recent messages
            channel_context: Context about the channel
            
        Returns:
            Decision object with fields:
            - should_respond: bool
            - reasoning: str
            - response_type: str
        """
        try:
            # Format messages for LLM
            formatted_messages = [
                f"{msg['author']}: {msg['content']}"
                for msg in messages[-5:]  # Only look at last 5 messages
            ]
            
            message_history = "\\n".join(formatted_messages)
            
            # Build prompt
            prompt = f'''
            Channel Context:
            {channel_context}
            
            Recent Messages:
            {message_history}
            
            Based on the conversation above, decide if you should respond.
            Consider:
            1. Is there a natural opening for you to join?
            2. Has someone asked a question you can help with?
            3. Is there an opportunity to add value to the conversation?
            
            Return a JSON object with:
            - should_respond (boolean)
            - reasoning (string)
            - response_type (string): "direct" for direct responses, "follow_up" for follow-up questions
            '''
            
            # Get decision from LLM
            response = await self._call_provider([{
                "role": "system",
                "content": prompt
            }])
            
            return response
            
        except Exception as e:
            print(f"Error in should_respond: {str(e)}")
            return {
                "should_respond": False,
                "reasoning": f"Error: {str(e)}",
                "response_type": None
            }

    async def _step(self, message: str) -> str:
        """
        Process a single conversation step.
        
        Args:
            message: The user's message
            
        Returns:
            Agent's response
        """
        # Add user message to history
        self.history.append({
            "role": "user",
            "content": message
        })
        
        # Get response from provider
        response = await self._call_provider(self.history)
        
        # Add response to history
        self.history.append({
            "role": "assistant",
            "content": response["content"]
        })
        
        return response["content"]

    async def process_new_messages(self, messages: List[Dict], channel_context: str) -> Optional[str]:
        """
        Process new messages and decide whether to respond.
        
        Args:
            messages: List of new messages
            channel_context: Context about the channel
            
        Returns:
            Optional response message
        """
        try:
            # Check cooldown
            if self.last_response:
                time_since_last = datetime.now() - self.last_response
                if time_since_last < self.cooldown:
                    return None
            
            # Get decision
            decision = await self.should_respond(messages, channel_context)
            
            if decision["should_respond"]:
                # Generate response
                response = await self._step(messages[-1]["content"])
                self.last_response = datetime.now()
                return response
                
            return None
            
        except Exception as e:
            print(f"Error processing messages: {str(e)}")
            return None

    async def run_behavior_loop(self):
        """Run the agent's behavior loop."""
        while True:
            try:
                await asyncio.sleep(self.scan_interval.total_seconds())
            except Exception as e:
                print(f"Error in behavior loop: {e}")
                continue 