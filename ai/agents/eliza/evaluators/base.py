"""
Base Evaluator Module

This module defines the base evaluator class that all evaluators must inherit from.
Evaluators are responsible for making decisions about agent behavior, such as:
- Whether to respond to a message
- How to process the response
- What actions to take
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseEvaluator(ABC):
    """
    Base class for all evaluators.
    
    Evaluators are responsible for making decisions about agent behavior.
    They take in state and context and return evaluation results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    @abstractmethod
    async def evaluate(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the given state and context.
        
        Args:
            state: Current state dictionary
            context: Context dictionary with additional information
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the evaluator."""
        return f"{self.__class__.__name__}(config={self.config})" 