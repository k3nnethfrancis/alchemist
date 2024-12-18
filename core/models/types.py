"""
types.py

This module contains shared type definitions and base classes used across the application.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class RuntimeConfig(BaseModel):
    """Configuration for the Agent Runtime"""
    agent_id: str
    model_provider: str
    model_name: str
    bot_user_id: Optional[int] = None


class MessageHistory(BaseModel):
    """Structure for storing message history"""
    user_id: int
    channel_id: int
    user_message: str
    agent_response: str
    timestamp: float