"""USING OPENAI IMPLEMENTATION"""
# """OpenPipe stream implementation."""

# from typing import Any, Dict, List, Optional, Generator
# from mirascope.core.base import BaseStream, BaseMessageParam, BaseTool
# from .response import OpenPipeCallResponse
# from .response_chunk import OpenPipeCallResponseChunk

# class OpenPipeStream(BaseStream):
#     """Stream implementation for OpenPipe."""
    
#     def __init__(
#         self,
#         response_chunks: List[OpenPipeCallResponseChunk],
#         tool_types: Optional[List[type[BaseTool]]] = None
#     ) -> None:
#         """Initialize stream with chunks and tool types."""
#         self.chunks = response_chunks
#         self.tool_types = tool_types or []
#         self._content = ""
#         self._usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        
#     @property
#     def content(self) -> str:
#         """Get accumulated content."""
#         return self._content
        
#     @property
#     def cost(self) -> float:
#         """Calculate stream cost."""
#         return 0.0  # OpenPipe doesn't charge per request
        
#     def _construct_message_param(
#         self, 
#         tool_calls: Optional[List[Dict[str, Any]]] = None,
#         content: Optional[str] = None
#     ) -> BaseMessageParam:
#         """Construct message parameter from stream data."""
#         return BaseMessageParam(
#             role="assistant",
#             content=content or self.content,
#             tool_calls=tool_calls
#         )
        
#     def construct_call_response(self) -> OpenPipeCallResponse:
#         """Construct final response from accumulated chunks."""
#         # Use the last chunk for metadata
#         last_chunk = self.chunks[-1] if self.chunks else None
        
#         return OpenPipeCallResponse(
#             id=last_chunk.id if last_chunk else "",
#             model=last_chunk.model if last_chunk else "",
#             usage=self._usage,
#             choices=[{
#                 "message": {
#                     "role": "assistant",
#                     "content": self.content
#                 },
#                 "finish_reason": last_chunk.finish_reasons[0] if last_chunk else "stop"
#             }],
#             tool_types=self.tool_types
#         )