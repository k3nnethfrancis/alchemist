"""Test suite for OpenPipe integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from mirascope.core import BaseMessageParam
from alchemist.core.mirascope.providers.openpipe import (
    openpipe_call,
    get_openpipe_client,
    OpenPipeCallResponse
)

@pytest.mark.asyncio
async def test_openpipe_client_creation():
    """Test OpenPipe client initialization."""
    with patch('alchemist.core.mirascope.providers.openpipe.client.OpenAI') as mock_openai:
        client = get_openpipe_client(api_key="test-key")
        mock_openai.assert_called_once_with(
            api_key="test-key",
            openpipe={
                "api_key": "test-key",
                "base_url": "https://api.openpipe.ai/api/v1"
            }
        )

def test_openpipe_call_decorator_sync():
    """Test OpenPipe call decorator synchronous functionality."""
    mock_response = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="Test response",
                    role="assistant"
                )
            )
        ]
    )

    with patch('alchemist.core.mirascope.providers.openpipe.client.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        @openpipe_call("gpt-4o-mini")
        def test_function(prompt: str) -> str:
            return prompt

        response = test_function("Hello")
        assert response.content == "Test response"
        
        # Verify OpenPipe client was used
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4o-mini"

@pytest.mark.asyncio
async def test_openpipe_call_decorator_async():
    """Test OpenPipe call decorator asynchronous functionality."""
    mock_response = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="Test response",
                    role="assistant"
                )
            )
        ]
    )

    with patch('alchemist.core.mirascope.providers.openpipe.client.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        @openpipe_call("gpt-4o-mini")
        async def test_function(prompt: str) -> str:
            return prompt

        response = await test_function("Hello")
        assert response.content == "Test response"

def test_openpipe_streaming_sync():
    """Test OpenPipe streaming functionality synchronously."""
    mock_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=" World"))])
    ]

    with patch('alchemist.core.mirascope.providers.openpipe.client.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=iter(mock_chunks))
        mock_openai.return_value = mock_client

        @openpipe_call("gpt-4o-mini", stream=True)
        def test_stream(prompt: str) -> str:
            return prompt

        stream = test_stream("Test")
        chunks = list(stream)
        assert len(chunks) == 2
        assert "".join(chunk.content for chunk, _ in chunks) == "Hello World"