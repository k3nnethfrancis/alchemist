"""The openpipe_call decorator for functions as LLM calls."""

from mirascope.core.base import call_factory
from mirascope.core.openai._utils import (
    get_json_output,
    handle_stream,
    handle_stream_async,
    setup_call,
)

from mirascope.core.openai.call_params import OpenAICallParams
from mirascope.core.openai.call_response import OpenAICallResponse
from mirascope.core.openai.call_response_chunk import OpenAICallResponseChunk
from mirascope.core.openai.stream import OpenAIStream
from mirascope.core.openai.tool import OpenAITool

# Create OpenPipe-specific configuration
class OpenPipeCallParams(OpenAICallParams):
    """Parameters for OpenPipe API calls."""
    stream: bool = False  # Disable streaming by default
    temperature: float = 0.7
    max_tokens: int = 1000

openpipe_call = call_factory(
    TCallResponse=OpenAICallResponse,
    TCallResponseChunk=OpenAICallResponseChunk,
    TToolType=OpenAITool,
    TStream=OpenAIStream,
    default_call_params=OpenPipeCallParams(),
    setup_call=setup_call,
    get_json_output=get_json_output,
    handle_stream=handle_stream,
    handle_stream_async=handle_stream_async,
)
"""A decorator for calling the OpenPipe API with a typed function."""