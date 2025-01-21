"""Tests for the image generation tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mirascope.core import BaseMessageParam
from alchemist.ai.tools.image import ImageGenerationTool

def test_tool_metadata():
    """Test tool name and description."""
    tool = ImageGenerationTool(prompt="test image")
    assert tool._name() == "generate_image"
    assert "dall-e" in tool._description().lower()
    assert "image" in tool._description().lower()

def test_validation():
    """Test parameter validation."""
    # Valid parameters
    tool = ImageGenerationTool(
        prompt="test image",
        size="1024x1024",
        quality="standard",
        model="dall-e-3"
    )
    assert tool.prompt == "test image"
    assert tool.size == "1024x1024"
    assert tool.quality == "standard"
    assert tool.model == "dall-e-3"
    
    # Invalid size should raise
    with pytest.raises(ValueError):
        ImageGenerationTool(prompt="test", size="invalid")
    
    # Invalid quality should raise
    with pytest.raises(ValueError):
        ImageGenerationTool(prompt="test", quality="invalid")
    
    # Invalid model should raise
    with pytest.raises(ValueError):
        ImageGenerationTool(prompt="test", model="invalid")

@pytest.mark.asyncio
async def test_image_generation():
    """Test image generation with mocked OpenAI client."""
    with patch("alchemist.ai.tools.image.openai_client") as mock_client:
        # Mock response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(url="https://test-image.url")]
        mock_client.images.generate = AsyncMock(return_value=mock_response)
        
        tool = ImageGenerationTool(
            prompt="A sunset over mountains",
            style="watercolor style",
            size="1024x1024",
            quality="standard",
            model="dall-e-3"
        )
        
        result = await tool.call()
        assert result == "https://test-image.url"
        
        # Verify DALL-E was called with correct parameters
        mock_client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="A sunset over mountains. watercolor style",
            size="1024x1024",
            quality="standard",
            n=1
        )

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling during image generation."""
    with patch("alchemist.ai.tools.image.openai_client") as mock_client:
        # Mock error response
        mock_client.images.generate = AsyncMock(
            side_effect=Exception("API error")
        )
        
        tool = ImageGenerationTool(prompt="test image")
        result = await tool.call()
        
        assert result.startswith("Error:")
        assert "API error" in result 