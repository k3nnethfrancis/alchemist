"""The Mirascope OpenPipe Module."""

from typing import TypeAlias
from openai.types.chat import ChatCompletionMessageParam
from mirascope.core.base import BaseMessageParam

# Since OpenPipe is OpenAI-compatible, we can use the same message type
OpenPipeMessageParam: TypeAlias = ChatCompletionMessageParam | BaseMessageParam

from mirascope.core.openai.call_response import OpenAICallResponse as OpenPipeCallResponse
from .call import openpipe_call

__all__ = [
    "OpenPipeCallResponse",
    "OpenPipeMessageParam",
    "openpipe_call",
]