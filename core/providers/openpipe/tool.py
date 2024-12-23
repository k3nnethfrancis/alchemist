"""USING OPENAI IMPLEMENTATION"""
# """OpenPipe tool implementation."""

# from typing import Any, Dict, Optional
# from pydantic import Field
# from mirascope.core.base import BaseTool
# from openai.types.chat import ChatCompletionMessageToolCall

# class OpenPipeTool(BaseTool):
#     """Base class for OpenPipe tools."""
    
#     __provider__ = "openpipe"
    
#     tool_call: ChatCompletionMessageToolCall
    
#     @classmethod
#     def tool_schema(cls) -> Dict[str, Any]:
#         """Tool schema for LLM."""
#         return {
#             "type": "function",
#             "function": {
#                 "name": cls._name(),
#                 "description": cls._description(),
#                 "parameters": cls.model_json_schema()
#             }
#         }
    
#     @classmethod
#     def from_tool_call(cls, tool_call: ChatCompletionMessageToolCall) -> "OpenPipeTool":
#         """Create tool instance from tool call."""
#         return cls.model_validate({
#             "tool_call": tool_call,
#             **cls._dict_from_json(tool_call.function.arguments)
#         })