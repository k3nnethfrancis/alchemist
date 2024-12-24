"""
Base Prompt Templates Module

This module provides the core prompt templates that can be customized for different
agent types. It includes:
- System prompt generation
- Persona integration
- Style guideline formatting
"""

from typing import Dict, List, Optional
from pydantic import BaseModel

from ai.prompts.persona import Persona

class PromptConfig(BaseModel):
    """Configuration for prompt generation."""
    include_personality: bool = True
    include_lore: bool = True
    include_style: bool = True
    extra_guidelines: List[str] = []

def create_system_prompt(persona: Persona, config: PromptConfig = PromptConfig()) -> Dict[str, str]:
    """
    Create a system prompt from a persona configuration.
    
    Args:
        persona (Persona): The persona to use
        config (PromptConfig): Configuration for prompt generation
        
    Returns:
        Dict[str, str]: System message configuration
    """
    sections = [f"You are {persona.name}"]
    if persona.nickname:
        sections[0] += f" ('{persona.nickname}')"
    sections[0] += f", {persona.bio}"
    
    if config.include_personality:
        sections.append("\nPersonality Traits:")
        for trait, value in persona.personality.traits.model_dump().items():
            sections.append(f"- {trait.title()}: {value:.1f}")
            
        sections.append("\nCore Stats:")
        for stat, value in persona.personality.stats.model_dump().items():
            sections.append(f"- {stat.title()}: {value:.1f}")
    
    if config.include_lore:
        sections.append("\nBackground:")
        for lore in persona.lore:
            sections.append(f"- {lore}")
    
    if config.include_style:
        sections.append("\nCommunication Style:")
        for style in persona.style.all:
            sections.append(f"- {style}")
        for style in persona.style.chat:
            sections.append(f"- {style}")
            
    if config.extra_guidelines:
        sections.append("\nAdditional Guidelines:")
        for guideline in config.extra_guidelines:
            sections.append(f"- {guideline}")
    
    return {
        "role": "system",
        "content": "\n".join(sections)
    }

def create_chat_prompt(persona: Persona) -> Dict[str, str]:
    """
    Create a chat-specific system prompt.
    
    Args:
        persona (Persona): The persona to use
        
    Returns:
        Dict[str, str]: System message configuration
    """
    config = PromptConfig(
        include_personality=True,
        include_lore=True,
        include_style=True,
        extra_guidelines=[
            "Focus on engaging, conversational responses",
            "Maintain consistent personality across interactions",
            "Use appropriate emoji and formatting for emphasis",
            "Keep responses concise but meaningful"
        ]
    )
    return create_system_prompt(persona, config) 