"""
Chat Agent Tools Module

This module provides tools for the ChatAgent, including:
- Image generation via DALL-E 3
- Future tools will be added here

Each tool follows Mirascope's BaseTool pattern and can be used with both
OpenAI and Anthropic providers.
"""

from typing import Literal
import logging
import asyncio
from openai import AsyncOpenAI
from mirascope.core import BaseTool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

class ImageGenerator(BaseTool):
    """
    Asynchronous image generation tool using DALL-E.
    
    Attributes:
        prompt (str): The prompt to generate an image from
    """
    
    prompt: str
    
    async def call(self) -> dict:
        """
        Generate an image asynchronously based on the provided prompt.
        
        Returns:
            dict: Contains status, url (if successful), and any relevant messages
        """
        try:
            client = AsyncOpenAI()
            
            logger.info(f"Generating image with prompt: {self.prompt}")
            
            response = await client.images.generate(
                model="dall-e-3",
                prompt=self.formatted_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            logger.info(f"Successfully generated image: {image_url}")
            
            return {
                "status": "success",
                "url": image_url,
                "prompt": self.prompt
            }
            
        except Exception as e:
            logger.error(f"Failed to generate image: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "prompt": self.prompt
            }

    @property
    def formatted_prompt(self) -> str:
        """Format the prompt with style guidelines."""
        return f"{self.prompt}. Blend the styles of Mobeus, solarpunk, and 70s sci-fi pulp."

# Example usage and testing
if __name__ == "__main__":
    from mirascope.core import BaseMessageParam, openai

    @openai.call("gpt-4o-mini", tools=[ImageGenerator])
    def generate_image(prompt: str) -> list[BaseMessageParam]:
        """Test function for image generation."""
        logger.info(f"Testing image generation with prompt: {prompt}")
        return [
            BaseMessageParam(
                role="user",
                content=f"Generate an image with the following prompt: {prompt}"
            )
        ]

    # Test the image generator
    test_prompt = "A beautiful furry cat in the sun"
    logger.info("Starting test run")
    response = generate_image(test_prompt)
    
    if tool := response.tool:
        logger.info("Tool detected in response, executing...")
        print(tool.call())
    else:
        logger.info("No tool in response, printing content")
        print(response.content)