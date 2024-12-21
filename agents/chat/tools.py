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

from mirascope.core import BaseTool
from openai import OpenAI
from pydantic import Field

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
    A tool for generating images using DALL-E 3. ALWAYS PROVIDE THE URL TO THE USER.
    
    Attributes:
        prompt (str): The description of the image to generate
    """

    prompt: str = Field(
        ...,
        description="Description of the image to generate"
    )

    def call(self) -> dict:
        """
        Generate an image using DALL-E 3.
        
        Returns:
            dict: Contains the image URL and generation status
        """
        logger.info(f"ImageGenerator.call() - Starting image generation with prompt: {self.prompt}")
        try:
            client = OpenAI()
            formatted_prompt = (
                f"{self.prompt}. "
                "Blend the styles of Mobeus, solarpunk, and 70s sci-fi pulp."
            )
            
            logger.debug(f"Sending request to DALL-E with formatted prompt: {formatted_prompt}")
            response = client.images.generate(
                model="dall-e-3",
                prompt=formatted_prompt,
                n=1,
            )
            
            url = response.data[0].url
            logger.info(f"Successfully generated image: {url}")
            return {
                "status": "success",
                "url": url,
                "prompt": self.prompt
            }
            
        except Exception as e:
            error_msg = f"Sorry, I couldn't generate that image: {str(e)}"
            logger.error(f"Image generation failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": error_msg,
                "prompt": self.prompt
            }


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