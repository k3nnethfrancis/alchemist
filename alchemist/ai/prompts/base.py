"""
Base system prompt module for chat agents.

This module provides a simple, modular system for creating chat prompts using Mirascope's
prompt templating system. It supports persona-based system messages and maintains
conversation history.
"""

from datetime import datetime
from typing import List, Optional

from mirascope.core import BaseMessageParam, Messages, prompt_template
from pydantic import BaseModel, Field

class PersonaConfig(BaseModel):
    """
    Configuration model for persona attributes.
    
    Attributes:
        id: Unique identifier for the persona
        name: Full name of the persona
        nickname: Optional nickname
        bio: Brief biography
        personality: Dictionary of personality traits and stats
        lore: List of background story elements
        style: Dictionary of style guidelines
    """
    id: str = Field(..., description="Unique identifier for the persona")
    name: str = Field(..., description="Full name of the persona")
    nickname: Optional[str] = Field(None, description="Optional nickname")
    bio: str = Field(..., description="Brief biography")
    personality: dict = Field(..., description="Dictionary of personality traits and stats")
    lore: List[str] = Field(..., description="List of background story elements")
    style: dict = Field(..., description="Dictionary of style guidelines")

@prompt_template()
def create_chat_prompt(
    system_content: str,
    query: str,
    history: Optional[List[BaseMessageParam]] = None
) -> List[BaseMessageParam]:
    """
    Creates a complete chat prompt with system, history, and user messages.
    
    Args:
        system_content: The system message content
        query: The user's query
        history: Optional conversation history
        
    Returns:
        List[BaseMessageParam]: Complete message list for the chat
    """
    messages = [BaseMessageParam(role="system", content=system_content)]
    
    if history:
        messages.extend(history)
        
    if query:
        messages.append(BaseMessageParam(role="user", content=query))
    
    return messages

def create_system_prompt(persona: PersonaConfig) -> str:
    """
    Creates a system prompt using the persona configuration.
    
    Args:
        persona: PersonaConfig object containing all persona attributes
        
    Returns:
        str: Formatted system message content
    """
    return f"""You are {persona.name}{f' - but you go by {persona.nickname} -' if persona.nickname else ','} {persona.bio}

The date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
###
{persona.name} embodies the following personality:
{persona.personality}
###
{persona.name} has the following lore:
{chr(10).join(f'- {item}' for item in persona.lore)}
###
{persona.name} has the following style:
{chr(10).join(f'- {item}' for item in persona.style['all'])}
###
You are entering a discord server to converse with your human friends. It's a chill day...
Begin simulation...
"""

## Test formatting
if __name__ == "__main__":
    from alchemist.ai.prompts.persona import AUG_E
    persona = PersonaConfig(**AUG_E)
    print(create_chat_prompt(create_system_prompt(persona), "Hi there!"))