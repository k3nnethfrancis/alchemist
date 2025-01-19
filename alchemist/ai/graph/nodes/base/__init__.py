"""Base node types for the graph system."""

from .llm import LLMNode
from .tool import ToolNode

__all__ = ["LLMNode", "ToolNode"]
