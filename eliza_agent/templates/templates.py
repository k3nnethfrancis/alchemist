"""
Discord Template Manager for Eliza AI Agent

This module manages the template generation for Discord interactions, combining
personality profiles with predefined prompt templates for different interaction types.

Templates include:
- Should Respond: Determines if the agent should respond to a message
- Voice Handler: Generates voice responses
- Message Handler: Generates text responses and actions

Author: Your Name
Date: Current Date
"""

from typing import Dict, Any
import yaml
from pathlib import Path

class DiscordTemplateManager:
    """
    Manages Discord-specific templates and combines them with personality profiles
    to generate appropriate prompts for different interaction types.
    """
    
    def __init__(self, template_path: str = "templates/discord.yaml"):
        """
        Initialize the Discord template manager.
        
        Args:
            template_path (str): Path to the YAML template file
        """
        self.templates = self._load_templates(template_path)
        self.message_completion_footer = "Assistant: I will respond as {{agentName}} in a natural, conversational way."
        self.should_respond_footer = "Result: "
        
    def _load_templates(self, template_path: str) -> Dict[str, str]:
        """
        Load templates from YAML file.
        
        Args:
            template_path (str): Path to template file
            
        Returns:
            Dict[str, str]: Dictionary of template names and their content
        """
        with open(template_path, 'r') as f:
            return yaml.safe_load(f)
            
    def get_should_respond_prompt(self, agent_profile: Dict[str, Any], recent_messages: str) -> str:
        """
        Generate the should-respond prompt for an agent.
        
        Args:
            agent_profile (Dict[str, Any]): Agent's personality profile
            recent_messages (str): Recent conversation history
            
        Returns:
            str: Formatted prompt for should-respond decision
        """
        template = self.templates['discord_should_respond']
        return template.format(
            agentName=agent_profile['name'],
            bio=agent_profile['bio'],
            recentMessages=recent_messages
        ) + self.should_respond_footer
