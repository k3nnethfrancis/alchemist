"""
Base system prompt module for chat agents.

This module defines the base system prompt template and formatting functions
using Mirascope's prompt templating system.
"""

from datetime import datetime
from mirascope.core import Messages, prompt_template
from pydantic import BaseModel

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
    id: str
    name: str
    nickname: str | None = None
    bio: str
    personality: dict
    lore: list[str]
    style: dict

@prompt_template()
def create_system_prompt(persona: PersonaConfig) -> dict:
    """
    Creates a system prompt using the persona configuration.
    
    Args:
        persona: PersonaConfig object containing all persona attributes
        
    Returns:
        dict: Formatted system message for the chat agent
    """
    nickname_part = f" - but you go by {persona.nickname} -" if persona.nickname else ","
    lore_formatted = "\n".join(f"- {item}" for item in persona.lore)
    style_formatted = "\n".join(f"- {item}" for item in persona.style['all'])
    
    system_content = f"""
You are {persona.name}{nickname_part} {persona.bio}

The date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
###
{persona.name} embodies the following personality:
{persona.personality}
###
###
{persona.name} has the following lore:
{lore_formatted}
###
###
{persona.name} has the following style:
{style_formatted}
###
You are entering a discord server to converse with your human friends. It's a chill day...
Begin simulation...
"""
    
    return {"role": "system", "content": system_content}


## Test formatting
if __name__ == "__main__":
    from alchemist.ai.prompts.persona import AUG_E
    persona = PersonaConfig(**AUG_E)
    print(create_system_prompt(persona))