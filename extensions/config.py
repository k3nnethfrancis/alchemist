"""
Extension Configuration Module

This module handles configuration management for different extensions.
It provides a centralized way to manage extension-specific settings and
credentials.

Usage:
    config = get_extension_config("discord")
"""

import os
import logging
import discord
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_discord_config() -> Dict[str, Any]:
    """
    Get Discord-specific configuration including token and intents.
    
    Returns:
        Dict[str, Any]: Configuration dictionary for Discord extension
        
    Raises:
        ValueError: If required environment variables are not set
    """
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
    
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    
    return {
        "token": token,
        "intents": intents
    }

def get_extension_config(extension: str) -> Dict[str, Any]:
    """
    Get configuration for specified extension.
    
    Args:
        extension (str): Name of the extension to configure
        
    Returns:
        Dict[str, Any]: Configuration dictionary for specified extension
        
    Raises:
        ValueError: If extension is not supported
    """
    if extension == "discord":
        return get_discord_config()
    
    raise ValueError(f"Unsupported extension: {extension}")
