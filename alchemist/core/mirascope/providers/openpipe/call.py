"""The openpipe_call decorator for functions as LLM calls."""

from typing import NotRequired
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
from .client import get_openpipe_client
import inspect

# Create OpenPipe-specific configuration
class OpenPipeCallParams(OpenAICallParams):
    """Parameters for OpenPipe API calls."""
    stream: bool = False  # Disable streaming by default
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]

def setup_openpipe_call(*args, **kwargs):
    """Wrapper around OpenAI's setup_call that ensures OpenPipe client is used."""
    if 'client' not in kwargs or kwargs['client'] is None:
        # Check if the function is async
        fn = kwargs.get('fn')
        if fn and inspect.iscoroutinefunction(fn):
            kwargs['client'] = get_openpipe_client()  # OpenPipe client is compatible with both sync/async
        else:
            kwargs['client'] = get_openpipe_client()
    return setup_call(*args, **kwargs)

openpipe_call = call_factory(
    TCallResponse=OpenAICallResponse,
    TCallResponseChunk=OpenAICallResponseChunk,
    TToolType=OpenAITool,
    TStream=OpenAIStream,
    default_call_params=OpenPipeCallParams(),
    setup_call=setup_openpipe_call,
    get_json_output=get_json_output,
    handle_stream=handle_stream,
    handle_stream_async=handle_stream_async,
)
"""A decorator for calling the OpenPipe API with a typed function."""