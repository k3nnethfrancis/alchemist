"""
Eliza Agent Package

A Discord-focused implementation of an Eliza-style agent using our graph system.
"""

from .agent import ElizaAgent
from .workflow import create_eliza_workflow

__all__ = ["ElizaAgent", "create_eliza_workflow"] 