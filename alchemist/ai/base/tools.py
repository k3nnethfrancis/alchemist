"""
Chat Agent Tools Module

This module provides tools for the ChatAgent, including:
- Image generation via DALL-E 3
- Future tools will be added here
"""

import logging
from typing import Optional
from openai import AsyncOpenAI
from mirascope.core import BaseTool
from pydantic import Field

# Configure logging
logger = logging.getLogger(__name__)

class ImageGenerationTool(BaseTool):
    """Tool for generating images using DALL-E 3."""
    
    prompt: str = Field(..., description="The description of the image to generate")
    style_guide: str = Field(
        default="Blend the styles of Mobeus, solarpunk, and 70s sci-fi pulp",
        description="Style guide for the image"
    )
    
    async def call(self) -> str:
        """
        Generate an image using DALL-E 3.
        
        Returns:
            str: URL of the generated image
        """
        try:
            logger.info(f"Generating image with prompt: {self.prompt}")
            
            # Format prompt with style guide
            formatted_prompt = f"{self.prompt}. {self.style_guide}"
            logger.debug(f"Formatted prompt: {formatted_prompt}")
            
            # Generate image using DALL-E
            client = AsyncOpenAI()
            response = await client.images.generate(
                model="dall-e-3",
                prompt=formatted_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            logger.info(f"Generated image URL: {image_url}")
            
            return image_url
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise


# test
if __name__ == "__main__":
    import asyncio
    tool = ImageGenerationTool(prompt="A spaceship going to the moon")
    print(asyncio.run(tool.call()))