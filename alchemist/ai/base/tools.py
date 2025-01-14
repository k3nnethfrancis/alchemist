"""
Chat Agent Tools Module

This module provides tools for the ChatAgent, including:
- Basic arithmetic via CalculatorTool
- Image generation via DALL-E 3
- Future tools will be added here

Each tool follows Mirascope's BaseTool pattern and includes:
- Proper inheritance from both BaseTool and BaseModel
- Clear docstrings and field descriptions
- Standalone test functionality
- Error handling
"""

import logging
import os
from typing import Optional
from openai import AsyncOpenAI
from mirascope.core import BaseTool
from pydantic import Field, BaseModel

# Configure logging
logger = logging.getLogger(__name__)

class CalculatorTool(BaseTool, BaseModel):
    """A simple calculator tool for basic arithmetic operations.
    
    This tool evaluates mathematical expressions using Python's eval() function.
    It supports basic arithmetic operations (+, -, *, /), exponents (**),
    and parentheses for grouping.
    
    Attributes:
        expression: The mathematical expression to evaluate
        
    Example:
        ```python
        tool = CalculatorTool(expression="2 + 2")
        result = tool.call()  # Returns "4"
        
        tool = CalculatorTool(expression="42 ** 0.5")
        result = tool.call()  # Returns "6.48074069840786"
        ```
    """
    
    expression: str = Field(
        ...,
        description="A mathematical expression to evaluate (e.g., '2 + 2', '42 ** 0.5')"
    )

    def call(self) -> str:
        """Evaluate the mathematical expression and return result.
        
        Returns:
            str: The result of the evaluation, or an error message if evaluation fails
            
        Example:
            >>> tool = CalculatorTool(expression="2 + 2")
            >>> tool.call()
            "4"
        """
        try:
            result = eval(self.expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

class ImageGenerationTool(BaseTool, BaseModel):
    """Tool for generating images using DALL-E 3."""
    
    prompt: str = Field(
        ...,
        description="The description of the image to generate",
        examples=["A spaceship going to the moon", "A cyberpunk city at sunset"]
    )
    style: str = Field(
        default="Blend the styles of Mobeus, solarpunk, and 70s sci-fi pulp",
        description="Style guide for the image generation"
    )

    @classmethod
    def _name(cls) -> str:
        return "generate_image"

    @classmethod
    def _description(cls) -> str:
        return "Generate images using DALL-E 3"

    async def call(self) -> str:
        """Generate an image using DALL-E 3."""
        try:
            # Format prompt with style guide
            formatted_prompt = f"{self.prompt}. {self.style}"
            logger.info(f"🎨 Generating image: {formatted_prompt}")
            
            # Generate image using DALL-E
            client = AsyncOpenAI()  # Uses API key from environment or client config
            response = await client.images.generate(
                model="dall-e-3",
                prompt=formatted_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            logger.info(f"✨ Generated image: {image_url}")
            
            return image_url
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {str(e)}")
            raise

# Test standalone tool functionality
if __name__ == "__main__":
    import asyncio
    
    async def test_tools():
        # Test Calculator
        print("\nTesting CalculatorTool:")
        calc = CalculatorTool(expression="42 ** 0.5")
        result = calc.call()
        print(f"√42 = {result}")
        
        # Test Image Generation
        print("\nTesting ImageGenerationTool:")
        try:
            image_tool = ImageGenerationTool(prompt="A spaceship going to the moon")
            result = await image_tool.call()
            print(f"Generated image URL: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    asyncio.run(test_tools())