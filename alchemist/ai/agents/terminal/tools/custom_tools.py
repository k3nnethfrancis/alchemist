"""
custom_tools.py

Defines custom tools for the agent.
"""

from typing import Any, ClassVar
from mirascope.core import BaseTool


class CustomTool(BaseTool):
    """
    A custom tool example.
    """

    name: ClassVar[str] = "custom_tool"
    description: ClassVar[str] = "A tool that performs a custom action."

    def _name(self) -> str:
        return self.name

    def call(self) -> Any:
        """
        Execute the tool's action.

        Returns:
            Any: The result of the tool's action.
        """
        # Implement the tool's functionality here
        return "CustomTool executed!"

class AnotherTool(BaseTool):
    another_attr: ClassVar[str] = "another_tool"
    
    def _name(self) -> str:
        return self.another_attr
    
    def call(self) -> Any:
        return "AnotherTool executed!"