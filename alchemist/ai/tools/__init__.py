"""Tools for AI agents.

This module provides a collection of tools that can be used by AI agents
to interact with various services and perform tasks.

Available tools:
- Calculator: Evaluate mathematical expressions
- Discord: Read and interact with Discord channels
- Image: Generate images using DALL-E
"""

from typing import List, Type
from mirascope.core import BaseTool

from .calculator import CalculatorTool
from .discord_tool import DiscordReaderTool
from .image import ImageGenerationTool

__all__: List[Type[BaseTool]] = [
    CalculatorTool,
    DiscordReaderTool,
    ImageGenerationTool
] 