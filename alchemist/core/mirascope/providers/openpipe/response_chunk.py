"""USING OPENAI IMPLEMENTATION"""
# """OpenPipe response implementation."""

# from typing import Any, Dict, List, Optional, Type
# import json
# from pydantic import Field, ConfigDict
# from mirascope.core.base import (
#     BaseTool, 
#     BaseCallResponse, 
#     BaseMessageParam,
#     transform_tool_outputs
# )

# class OpenPipeCallResponse(BaseCallResponse):
#     """Response from an OpenPipe call."""
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)
    
#     response: Any = Field(description="Raw OpenPipe response")
#     metadata: Dict[str, Any] = Field(default_factory=dict)
#     tool_types: List[Type[BaseTool]] = Field(default_factory=list)
    
#     @property
#     def id(self) -> str:
#         """Get response ID."""
#         return getattr(self.response, "id", "")
    
#     @property
#     def model(self) -> str:
#         """Get model name."""
#         return getattr(self.response, "model", "")
    
#     @property
#     def usage(self) -> Dict[str, int]:
#         """Get usage statistics."""
#         return getattr(self.response, "usage", {})
    
#     @property
#     def input_tokens(self) -> int:
#         """Get number of input tokens."""
#         return self.usage.get("prompt_tokens", 0)
    
#     @property
#     def output_tokens(self) -> int:
#         """Get number of output tokens."""
#         return self.usage.get("completion_tokens", 0)
    
#     @property
#     def cost(self) -> float:
#         """Get cost of the request."""
#         return 0.0  # OpenPipe doesn't charge per request
    
#     @property
#     def content(self) -> str:
#         """Get response content."""
#         choices = getattr(self.response, "choices", [])
#         if not choices:
#             return ""
#         message = choices[0].message
#         return message.content if message else ""

#     @property
#     def message_param(self) -> BaseMessageParam:
#         """Get message parameter."""
#         choices = getattr(self.response, "choices", [])
#         if not choices:
#             return BaseMessageParam(role="assistant", content="")
#         message = choices[0].message
#         return BaseMessageParam(
#             role=message.role,
#             content=message.content if message else "",
#             tool_calls=getattr(message, "tool_calls", None)
#         )

#     @property
#     def finish_reasons(self) -> List[str]:
#         """Get finish reasons from all choices."""
#         choices = getattr(self.response, "choices", [])
#         return [choice.finish_reason for choice in choices] if choices else ["stop"]

#     @property
#     def tools(self) -> List[BaseTool]:
#         """Get all tools from the response."""
#         if not self.response.choices:
#             return []
            
#         message = self.response.choices[0].message
#         tool_calls = getattr(message, "tool_calls", [])
        
#         if not tool_calls or not self.tool_types:
#             return []
            
#         tools = []
#         for tool_call in tool_calls:
#             tool_name = tool_call.function.name
            
#             # Find matching tool type
#             tool_type = next(
#                 (t for t in self.tool_types if t.__name__.lower() == tool_name.lower()),
#                 None
#             )
            
#             if not tool_type:
#                 continue
                
#             try:
#                 # Parse arguments and create tool instance
#                 args = json.loads(tool_call.function.arguments)
#                 tool = tool_type(**args)
#                 tool.tool_call = tool_call  # Store original tool call
#                 tools.append(tool)
#             except Exception as e:
#                 logger.error(f"Error creating tool: {e}")
#                 continue
                
#         return tools

#     @property
#     def tool(self) -> Optional[BaseTool]:
#         """Get the first tool if present."""
#         tools = self.tools
#         return tools[0] if tools else None

#     @transform_tool_outputs
#     def tool_message_params(
#         self, tools_and_outputs: list[tuple[BaseTool, str]]
#     ) -> list[Dict[str, Any]]:
#         """Returns the tool message parameters for tool call results."""
#         messages = []
#         for tool, output in tools_and_outputs:
#             tool_id = f"call_{tool.__class__.__name__.lower()}"
            
#             # Assistant message with tool call
#             messages.append({
#                 "role": "assistant",
#                 "content": None,
#                 "tool_calls": [{
#                     "id": tool_id,
#                     "type": "function",
#                     "function": {
#                         "name": tool.__class__.__name__,
#                         "arguments": tool.model_dump_json()
#                     }
#                 }]
#             })
            
#             # Tool response message
#             messages.append({
#                 "role": "tool",
#                 "content": str(output),
#                 "tool_call_id": tool_id
#             })
            
#         return messages
# """OpenPipe response chunk implementation."""

# from typing import Any, Dict, List, Optional
# from mirascope.core.base import BaseCallResponseChunk, BaseMessageParam
# from pydantic import Field, ConfigDict

# class OpenPipeCallResponseChunk(BaseCallResponseChunk):
#     """Chunk response from an OpenPipe streaming call."""
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)
    
#     # Required fields
#     response_id: str = Field(alias="id")
#     response_model: str = Field(alias="model")
#     response_usage: Dict[str, Any] = Field(default_factory=dict, alias="usage")
    
#     # OpenPipe specific fields
#     object: str = Field(default="chat.completion.chunk")
#     created: int = Field(default=0)
#     choices: List[Dict[str, Any]] = Field(default_factory=list)
    
#     @property
#     def id(self) -> str:
#         """Get response ID."""
#         return self.response_id
    
#     @property
#     def model(self) -> str:
#         """Get model name."""
#         return self.response_model
    
#     @property
#     def usage(self) -> Dict[str, Any]:
#         """Get usage statistics."""
#         return self.response_usage
    
#     @property
#     def content(self) -> str:
#         """Get content from the first choice."""
#         if not self.choices:
#             return ""
#         return self.choices[0].get("delta", {}).get("content") or ""
    
#     @property
#     def finish_reasons(self) -> List[str]:
#         """Get finish reasons from all choices."""
#         return [choice.get("finish_reason", None) for choice in self.choices] 