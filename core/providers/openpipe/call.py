"""The openpipe_call decorator for functions as LLM calls."""

from mirascope.core.base import call_factory
from mirascope.core.openai._utils import (
    get_json_output,
    handle_stream,
    handle_stream_async,
    setup_call,
)
# from .call_params import OpenPipeCallParams
from mirascope.core.openai.call_params import OpenAICallParams as OpenPipeCallParams
# from .response import OpenPipeCallResponse
# from .response_chunk import OpenPipeCallResponseChunk
# from .stream import OpenPipeStream
# from .tool import OpenPipeTool

from mirascope.core.openai.call_response import OpenAICallResponse as OpenPipeCallResponse
from mirascope.core.openai.call_response_chunk import OpenAICallResponseChunk as OpenPipeCallResponseChunk
from mirascope.core.openai.stream import OpenAIStream as OpenPipeStream
from mirascope.core.openai.tool import OpenAITool as OpenPipeTool

openpipe_call = call_factory(
    TCallResponse=OpenPipeCallResponse,
    TCallResponseChunk=OpenPipeCallResponseChunk,
    TToolType=OpenPipeTool,
    TStream=OpenPipeStream,
    default_call_params=OpenPipeCallParams(),
    setup_call=setup_call,
    get_json_output=get_json_output,
    handle_stream=handle_stream,
    handle_stream_async=handle_stream_async,
)
"""A decorator for calling the OpenPipe API with a typed function."""