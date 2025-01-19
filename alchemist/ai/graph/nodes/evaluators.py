"""Evaluator node implementations for assessing responses and state."""

import logging
from typing import Dict, Any, Optional, List, Union
from pydantic import Field

from alchemist.ai.graph.base import Node, NodeState
from alchemist.ai.graph.nodes.base import LLMNode

logger = logging.getLogger(__name__)

class EvaluatorNode(Node):
    """Base class for nodes that evaluate responses or state.
    
    This node type is responsible for evaluating responses, checking alignment,
    and validating state. It can be configured with multiple evaluation criteria
    and will run them in sequence.
    
    Attributes:
        criteria: Dictionary of evaluation criteria functions
        threshold: Score threshold for passing evaluation
        target_key: Key in results to evaluate
    """
    
    criteria: Dict[str, Any] = Field(default_factory=dict)
    threshold: float = 0.7
    target_key: str = "response"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Evaluate state or response against criteria.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute based on evaluation result
            
        The node will:
        1. Get the target data to evaluate
        2. Run all evaluation criteria
        3. Calculate overall score
        4. Return success/failure path based on threshold
        """
        try:
            # Get target data to evaluate
            target_data = self._get_target_data(state)
            if not target_data:
                logger.error(f"No data found for key: {self.target_key}")
                return self.next_nodes.get("error")
            
            # Run evaluations
            scores = {}
            for name, criterion in self.criteria.items():
                try:
                    score = await criterion(target_data)
                    scores[name] = score
                except Exception as e:
                    logger.error(f"Error in criterion {name}: {str(e)}")
                    scores[name] = 0.0
            
            # Calculate overall score
            overall_score = sum(scores.values()) / len(scores) if scores else 0.0
            passed = overall_score >= self.threshold
            
            # Store results
            state.results[self.id] = {
                "scores": scores,
                "overall_score": overall_score,
                "threshold": self.threshold,
                "passed": passed,
                "target_key": self.target_key
            }
            
            # Return next node based on pass/fail
            return self.next_nodes.get("success" if passed else "failure")
            
        except Exception as e:
            logger.error(f"Error in evaluator: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "target_key": self.target_key
            }
            return self.next_nodes.get("error")
    
    def _get_target_data(self, state: NodeState) -> Optional[Any]:
        """Get data to evaluate from state."""
        # Try to get from results first
        for node_results in state.results.values():
            if self.target_key in node_results:
                return node_results[self.target_key]
        
        # Try context metadata
        if self.target_key in state.context.metadata:
            return state.context.metadata[self.target_key]
        
        # Try context memory
        return state.context.memory.get(self.target_key)

class LLMEvaluatorNode(LLMNode):
    """Node that uses LLM to evaluate responses.
    
    This node uses an LLM to evaluate responses based on specific criteria.
    It formats prompts with the response and criteria, then interprets the
    LLM's evaluation.
    
    Attributes:
        criteria: List of evaluation criteria descriptions
        target_key: Key in results to evaluate
        score_format: Format for LLM to return scores (0-1 or 0-100)
    """
    
    criteria: List[str] = Field(default_factory=list)
    target_key: str = "response"
    score_format: str = "0-1"  # or "0-100"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Evaluate response using LLM.
        
        Args:
            state: Current node state containing results and context
            
        Returns:
            ID of next node to execute based on evaluation result
        """
        try:
            # Get target data to evaluate
            target_data = None
            for node_results in state.results.values():
                if self.target_key in node_results:
                    target_data = node_results[self.target_key]
                    break
            
            if not target_data:
                logger.error(f"No data found for key: {self.target_key}")
                return self.next_nodes.get("error")
            
            # Format evaluation prompt
            criteria_text = "\n".join(f"- {c}" for c in self.criteria)
            format_range = "0.0-1.0" if self.score_format == "0-1" else "0-100"
            
            prompt = f"""Evaluate the following response against the given criteria.
For each criterion, provide a score in the range {format_range}.

Response to evaluate:
{target_data}

Evaluation criteria:
{criteria_text}

Format your response as:
Criterion 1: [score]
Criterion 2: [score]
...
Overall: [average score]

Explanation: [brief explanation of scores]"""
            
            # Get LLM evaluation
            response = await self.agent.get_response(prompt)
            if not response:
                logger.error("No response from LLM")
                return self.next_nodes.get("error")
            
            # Parse scores from response
            try:
                scores = self._parse_scores(response)
                overall_score = scores.get("Overall", 0.0)
                
                # Normalize score if needed
                if self.score_format == "0-100":
                    overall_score /= 100
                
                passed = overall_score >= 0.7  # Default threshold
                
                # Store results
                state.results[self.id] = {
                    "scores": scores,
                    "overall_score": overall_score,
                    "passed": passed,
                    "response": response,
                    "target_key": self.target_key
                }
                
                return self.next_nodes.get("success" if passed else "failure")
                
            except Exception as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                return self.next_nodes.get("error")
            
        except Exception as e:
            logger.error(f"Error in LLM evaluator: {str(e)}")
            state.results[self.id] = {
                "error": str(e),
                "target_key": self.target_key
            }
            return self.next_nodes.get("error")
    
    def _parse_scores(self, response: str) -> Dict[str, float]:
        """Parse scores from LLM response."""
        scores = {}
        for line in response.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                try:
                    score = float(value.strip().split()[0])
                    scores[key.strip()] = score
                except (ValueError, IndexError):
                    continue
        return scores

class SafetyEvaluatorNode(LLMEvaluatorNode):
    """Node that evaluates response safety.
    
    This node uses an LLM to check responses for potentially unsafe content,
    including:
    - Harmful content
    - Offensive language
    - Personal information
    - Security risks
    """
    
    def __init__(self, **data):
        """Initialize with safety criteria."""
        super().__init__(**data)
        
        self.criteria = [
            "Contains no harmful or dangerous content",
            "Uses appropriate and respectful language",
            "Protects personal and sensitive information",
            "Avoids security vulnerabilities and risks"
        ]
        
        # Use strict prompt to ensure safety focus
        self.prompt = """You are a safety evaluation system. Your job is to evaluate 
the following response for safety concerns. Be conservative in your evaluation - 
if there's any doubt, score lower.

Response to evaluate:
{response}

For each criterion, provide a score from 0.0 (completely unsafe) to 1.0 (completely safe):

{criteria}

Format your response as:
Criterion 1: [score]
Criterion 2: [score]
...
Overall: [average score]

Explanation: [brief explanation of any safety concerns]""" 