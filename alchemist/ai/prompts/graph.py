"""Graph Node Prompts

This module provides prompt templates for graph nodes:
1. Decision prompts - For making structured choices
2. Action prompts - For tool execution
3. Response prompts - For generating output
"""

from typing import Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field

from alchemist.ai.prompts.persona import Persona

class NodeType(str, Enum):
    """Types of graph nodes."""
    DECISION = "decision"
    ACTION = "action"
    RESPONSE = "response"

class GraphPromptConfig(BaseModel):
    """Configuration for graph prompt generation."""
    node_type: NodeType
    options: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    style_guidelines: List[str] = Field(default_factory=list)
    extra_context: Dict[str, Any] = Field(default_factory=dict)

class DecisionPrompt(BaseModel):
    """Template for decision nodes."""
    context: Dict[str, Any]
    choices: List[str]
    examples: List[Dict[str, Any]]
    prompt: str
    
    def format(self) -> str:
        """Format the prompt with context."""
        return f"""Given the following context and examples, make a decision by selecting one of the available choices.

Context:
{self.context}

Available Choices:
{self.choices}

Examples:
{self.examples}

Decision Task:
{self.prompt}

Your response should be exactly one of the available choices."""

class ActionPrompt(BaseModel):
    """Template for action nodes."""
    context: Dict[str, Any]
    tools: List[Dict[str, Any]]
    prompt: str
    
    def format(self) -> str:
        """Format the prompt with context."""
        return f"""Given the following context and available tools, determine how to execute the action.

Context:
{self.context}

Available Tools:
{self.tools}

Action Task:
{self.prompt}

Your response should be a valid tool execution plan."""

class ResponsePrompt(BaseModel):
    """Template for response nodes."""
    context: Dict[str, Any]
    examples: List[Dict[str, Any]]
    prompt: str
    format_instructions: str = ""
    
    def format(self) -> str:
        """Format the prompt with context."""
        return f"""Generate a response based on the following context and requirements.

Context:
{self.context}

Examples:
{self.examples}

Response Task:
{self.prompt}

{self.format_instructions}

Your response should follow the provided format and guidelines."""

def create_decision_prompt(persona: Persona, config: GraphPromptConfig) -> Dict[str, str]:
    """Create a decision prompt with persona context."""
    context = {
        "persona": persona.name,
        "personality": persona.personality.traits.model_dump(),
        **config.extra_context
    }
    
    prompt = DecisionPrompt(
        context=context,
        choices=config.options,
        examples=[],  # Could be populated from persona examples
        prompt=f"As {persona.name}, make a decision between the available choices."
    )
    
    return {
        "role": "system",
        "content": prompt.format()
    }

def create_action_prompt(
    persona: Persona,
    config: GraphPromptConfig,
    state: Dict[str, Any]
) -> Dict[str, str]:
    """Create an action prompt with state context."""
    context = {
        "persona": persona.name,
        "state": state,
        **config.extra_context
    }
    
    prompt = ActionPrompt(
        context=context,
        tools=[{"name": tool} for tool in config.tools],
        prompt=f"As {persona.name}, determine which tool to use and how to execute it."
    )
    
    return {
        "role": "system",
        "content": prompt.format()
    }

def create_response_prompt(
    persona: Persona,
    config: GraphPromptConfig,
    state: Dict[str, Any]
) -> Dict[str, str]:
    """Create a response prompt with style guidelines."""
    context = {
        "persona": persona.name,
        "state": state,
        **config.extra_context
    }
    
    prompt = ResponsePrompt(
        context=context,
        examples=[],  # Could be populated from persona examples
        prompt=f"As {persona.name}, generate a response following the style guidelines.",
        format_instructions="\n".join(
            f"- {guideline}" for guideline in config.style_guidelines
        )
    )
    
    return {
        "role": "system",
        "content": prompt.format()
    } 