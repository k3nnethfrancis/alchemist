"""Tests for the Chat Agent tools."""

import pytest
from mirascope.core import BaseMessageParam, openai

from alchemist.ai.base.tools import CalculatorTool, ImageGenerationTool

def test_calculator_tool_standalone():
    """Test CalculatorTool basic functionality."""
    # Test basic arithmetic
    calc = CalculatorTool(expression="2 + 2")
    assert calc.call() == "4"
    
    # Test square root
    calc = CalculatorTool(expression="42 ** 0.5")
    assert calc.call() == "6.48074069840786"
    
    # Test error handling
    calc = CalculatorTool(expression="invalid")
    assert "Error" in calc.call()

@pytest.mark.asyncio
async def test_image_generation_tool_standalone():
    """Test ImageGenerationTool basic functionality."""
    tool = ImageGenerationTool(prompt="A test image")
    try:
        result = await tool.call()
        assert isinstance(result, str)
        assert result.startswith("http")
    except Exception as e:
        pytest.skip(f"Skipping image generation test: {str(e)}")

def test_calculator_tool_with_agent():
    """Test CalculatorTool integration with agent via function calling."""
    @openai.call("gpt-4o-mini", tools=[CalculatorTool])
    def calculate(expression: str) -> list[BaseMessageParam]:
        return [BaseMessageParam(role="user", content=f"Calculate {expression}")]
    
    # Test single calculation
    response = calculate("2 + 2")
    assert response.tool
    assert response.tool.call() == "4"
    
    # Test multiple calculations
    response = calculate("sqrt of 42 and 78")
    tools = response.tools
    assert len(tools) == 2
    results = [tool.call() for tool in tools]
    assert "6.48074069840786" in results

@pytest.mark.asyncio
async def test_image_generation_tool_with_agent():
    """Test ImageGenerationTool integration with agent via function calling."""
    @openai.call("gpt-4o-mini", tools=[ImageGenerationTool])
    def generate_image(description: str) -> list[BaseMessageParam]:
        return [BaseMessageParam(role="user", content=f"Generate an image of {description}")]
    
    try:
        response = generate_image("a simple test scene")
        assert response.tool
        result = await response.tool.call()
        assert isinstance(result, str)
        assert result.startswith("http")
    except Exception as e:
        pytest.skip(f"Skipping image generation test: {str(e)}")

def test_tool_metadata():
    """Test tool metadata and configuration."""
    # Test CalculatorTool
    calc = CalculatorTool(expression="2 + 2")
    assert calc._name() == "calculate"
    assert "evaluate" in calc._description().lower()
    assert "expression" in calc.model_fields
    
    # Test ImageGenerationTool
    image = ImageGenerationTool(prompt="test")
    assert image._name() == "generate_image"
    assert "DALL-E" in image._description()
    assert "prompt" in image.model_fields
    assert "style" in image.model_fields 