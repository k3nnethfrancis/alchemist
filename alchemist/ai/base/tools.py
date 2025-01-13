"""
Chat Agent Tools Module

This module provides tools for the ChatAgent, including:
- Image generation via DALL-E 3
- Future tools will be added here
"""

import logging
from typing import Optional, ClassVar
from openai import AsyncOpenAI
from mirascope.core import BaseTool
from pydantic import Field
import discord
from datetime import datetime, timedelta

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

class DiscordReaderTool(BaseTool):
    """Tool for reading messages from Discord channels."""
    
    name = "read_discord_messages"
    description = "Reads and summarizes recent messages from a Discord channel"
    
    def __init__(self, client: discord.Client):
        self.client = client
        super().__init__()
    
    async def _run(self, channel_name: str, days: int = 2) -> str:
        """Read messages from a Discord channel."""
        try:
            # Find channel by name
            channel = discord.utils.get(self.client.get_all_channels(), name=channel_name)
            if not channel:
                return f"Channel '{channel_name}' not found"
            
            # Use existing client method
            messages = await self.client.fetch_channel_history(
                channel_id=channel.id,
                days=days
            )
            
            return f"Found {len(messages)} messages:\n" + "\n".join(
                f"{msg['author']['name']} ({datetime.fromtimestamp(msg['timestamp']).strftime('%Y-%m-%d %H:%M')}): {msg['content']}"
                for msg in messages
            )
            
        except Exception as e:
            logger.error(f"Failed to read Discord messages: {str(e)}")
            raise
    
    async def __call__(self, channel_name: str, days: int = 2) -> str:
        return await self._run(channel_name, days)


# test
if __name__ == "__main__":
    import asyncio
    tool = ImageGenerationTool(prompt="A spaceship going to the moon")
    print(asyncio.run(tool.call()))