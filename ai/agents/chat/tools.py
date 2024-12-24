"""
Chat Agent Tools Module

This module defines tools that can be used by the ChatAgent.
"""

import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from mirascope.core.base import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)

class ImageGenerationTool(BaseTool):
    """
    Tool for generating images using OpenAI's DALL-E API.
    
    Attributes:
        prompt (str): The text prompt to generate an image from
        style_guide (Optional[str]): Optional style guide for the image
    """
    
    prompt: str
    style_guide: Optional[str] = None
    
    async def call(self) -> str:
        """
        Call the image generation API and return the URL of the generated image.
        
        Returns:
            str: URL of the generated image
            
        Raises:
            Exception: If image generation fails due to content policy or other errors
        """
        try:
            # Format prompt with style guide if provided
            formatted_prompt = self.prompt
            if self.style_guide:
                formatted_prompt = f"{self.prompt}. {self.style_guide}"
            
            logger.info(f"Generating image with prompt: {formatted_prompt}")
            
            # Initialize OpenAI client
            client = AsyncOpenAI()
            
            # Call OpenAI's image generation API
            response = await client.images.generate(
                model="dall-e-3",
                prompt=formatted_prompt,
                n=1,
                size="1024x1024"
            )
            
            image_url = response.data[0].url
            logger.info(f"Successfully generated image: {image_url}")
            return image_url
            
        except Exception as e:
            error_msg = str(e)
            if "content_policy_violation" in error_msg.lower() or "safety" in error_msg.lower():
                error = (
                    f"I couldn't generate that image due to content policy restrictions. "
                    f"The prompt was: '{self.prompt}'. "
                    "Could you try a different description that follows OpenAI's safety guidelines?"
                )
            else:
                error = (
                    f"I encountered an issue while generating the image: {error_msg}. "
                    "Would you like to try again with a different prompt?"
                )
            
            logger.error(error)
            raise Exception(error)
    
    def format_response(self, response: str, result: str) -> str:
        """
        Format the response to include the generated image URL if needed.
        
        Args:
            response (str): The follow-up response from the agent
            result (str): The image URL from the tool call
            
        Returns:
            str: The formatted response with image URL
        """
        # If the URL is already in the response, just return it as is
        if result in response:
            return response
            
        # Otherwise, append the image markdown
        return f"{response}\\n\\n![generated image]({result})"
    
    @classmethod
    def from_args(cls, args: Dict[str, Any]) -> "ImageGenerationTool":
        """
        Create a tool instance from arguments.
        
        Args:
            args (Dict[str, Any]): Tool arguments
            
        Returns:
            ImageGenerationTool: New tool instance
        """
        return cls(
            prompt=args.get("prompt", "A mysterious image"),
            style_guide=args.get("style_guide", cls.style_guide.default)
        ) 