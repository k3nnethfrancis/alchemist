"""
Message Evaluator Module

This module provides the message evaluation functionality using Mirascope.
It determines whether an agent should respond to a message based on:
- Message content
- Conversation context
- Agent's character profile
- Channel activity
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from mirascope import Messages, prompt_template, openai
from .base import BaseEvaluator

logger = logging.getLogger(__name__)

class MessageEvaluator(BaseEvaluator):
    """
    Evaluates messages to determine if and how an agent should respond.
    
    This evaluator:
    1. Checks if enough time has passed since last response
    2. Evaluates message relevance to agent's character
    3. Determines appropriate response type
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the message evaluator.
        
        Args:
            config: Configuration with optional keys:
                - min_delay: Minimum seconds between responses (default: 30)
                - max_history: Max messages to include in context (default: 10)
        """
        super().__init__(config)
        self.min_delay = self.config.get('min_delay', 30)
        self.max_history = self.config.get('max_history', 10)
        self._last_response = {}  # channel_id -> datetime
        
    def _format_context(self, messages: List[Dict[str, Any]]) -> str:
        """Format message history into a string context."""
        return "\n".join(
            f"{msg['author']}: {msg['content']}" 
            for msg in messages[-self.max_history:]
        )
        
    def _should_check_timing(self, channel_id: str) -> bool:
        """Check if enough time has passed since last response."""
        if channel_id not in self._last_response:
            return True
            
        elapsed = datetime.utcnow() - self._last_response[channel_id]
        return elapsed.total_seconds() >= self.min_delay
        
    @openai.call("gpt-4o-mini")
    @prompt_template("""
    You are {name}, {bio}
    
    Recent conversation:
    {context}
    
    Current message: {message}
    
    Evaluate if you should respond based on:
    1. Message relevance to your character
    2. Conversation context and flow
    3. Value you can add to the discussion
    4. Your character's interests and expertise
    
    Respond with a JSON object:
    {
        "should_respond": true/false,
        "confidence": 0-1 float,
        "reasoning": "Brief explanation",
        "response_type": "normal" | "question" | "action" | "ignore"
    }
    """)
    async def _evaluate_message(
        self,
        name: str,
        bio: str,
        context: str,
        message: str
    ) -> Dict[str, Any]:
        """Evaluate a single message using the LLM."""
        return {}
        
    async def evaluate(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate whether to respond to a message.
        
        Args:
            state: Current state containing:
                - channel_id: Channel identifier
                - message: Current message content
                - author: Message author
                - character: Agent character info
                - history: List of recent messages
            context: Additional context (unused currently)
            
        Returns:
            Evaluation results including:
            - should_respond: Whether to respond
            - confidence: Confidence in the decision
            - reasoning: Explanation of the decision
            - response_type: Type of response to generate
        """
        try:
            channel_id = state['channel_id']
            
            # Check timing
            if not self._should_check_timing(channel_id):
                return {
                    'should_respond': False,
                    'confidence': 1.0,
                    'reasoning': 'Too soon since last response',
                    'response_type': 'ignore'
                }
                
            # Format context
            context_str = self._format_context(state['history'])
            
            # Evaluate with LLM
            result = await self._evaluate_message(
                name=state['character']['name'],
                bio=state['character']['bio'],
                context=context_str,
                message=state['message']
            )
            
            # Update timing if should respond
            if result.get('should_respond', False):
                self._last_response[channel_id] = datetime.utcnow()
                
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating message: {str(e)}")
            return {
                'should_respond': False,
                'confidence': 1.0,
                'reasoning': f'Error: {str(e)}',
                'response_type': 'ignore'
            } 