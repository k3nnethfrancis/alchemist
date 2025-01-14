"""
Chat Agent Tools Module

This module provides tools for the ChatAgent, including:
- Basic arithmetic via CalculatorTool
- Image generation via DALL-E 3
- Discord channel reading via DiscordReaderTool
- Future tools will be added here

Each tool follows Mirascope's BaseTool pattern and includes:
- Proper inheritance from both BaseTool and BaseModel
- Clear docstrings and field descriptions
- Standalone test functionality
- Error handling
"""

import logging
import os
from typing import Optional, Dict, Tuple, Type
from openai import AsyncOpenAI
from mirascope.core import BaseTool
from pydantic import Field, BaseModel
import aiohttp
from datetime import datetime, timezone, timedelta

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

class DiscordReaderTool(BaseTool, BaseModel):
    """Tool for reading Discord channel history.
    
    This tool requires a separate Discord bot running with read permissions.
    The bot should be started using core/discord/run_reader_bot.py before using this tool.
    
    Attributes:
        channel: The Discord channel to read from (can be ID or name)
        lookback: The time period to look back (e.g., "1h", "2d", "30m")
    """
    
    channel: str = Field(
        ...,
        description="The Discord channel to read from (can be channel ID or name like 'general')"
    )
    lookback: str = Field(
        ...,
        description="Time period to look back (e.g., '1h', '2d', '30m')",
        pattern=r'^\d+[hdm]$'  # Validates format like "1h", "2d", "30m"
    )
    
    # Class-level channel mapping
    _channel_map: Dict[str, str] = {}
    
    @classmethod
    def configure(cls, channel_map: Dict[str, str]) -> None:
        """Configure the tool with channel mappings.
        
        Args:
            channel_map: Mapping of channel names to channel IDs
        """
        cls._channel_map = channel_map

    @classmethod
    def _name(cls) -> str:
        return "read_discord"

    @classmethod
    def _description(cls) -> str:
        channel_list = ", ".join(f"'{name}'" for name in cls._channel_map.keys())
        return f"Read message history from a Discord channel. Available channels: {channel_list}"

    def _parse_lookback(self) -> Tuple[int, str]:
        """Parse lookback string into amount and unit."""
        amount = int(self.lookback[:-1])
        unit = self.lookback[-1]
        return amount, unit

    def _convert_to_days(self, amount: int, unit: str) -> float:
        """Convert lookback to days for the Discord client."""
        if unit == 'd':
            return float(amount)
        elif unit == 'h':
            return amount / 24.0
        elif unit == 'm':
            return amount / (24.0 * 60)
        raise ValueError(f"Invalid time unit: {unit}")

    def _resolve_channel_id(self) -> str:
        """Resolve channel name to ID if needed."""
        # If it's already an ID, return it
        if self.channel.isdigit():
            return self.channel
            
        # Remove # prefix if present
        channel_name = self.channel.lstrip('#')
        
        # Look up in channel map
        if channel_id := self._channel_map.get(channel_name):
            return channel_id
            
        raise ValueError(f"Unknown channel: {self.channel}. Available channels: {', '.join(self._channel_map.keys())}")

    async def call(self) -> str:
        """Read Discord channel history.
        
        Returns:
            str: Formatted message history or error message
        """
        try:
            # Resolve channel ID
            channel_id = self._resolve_channel_id()
            
            # Parse lookback period
            amount, unit = self._parse_lookback()
            days = self._convert_to_days(amount, unit)
            
            # Connect to Discord reader service
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:5000/read_channel",
                    params={
                        "channel_id": channel_id,
                        "days": days
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Failed to read channel: {error_text}")
                    
                    messages = await response.json()
                    
                    if not messages:
                        return "No messages found in the specified time period."
                    
                    # Format messages nicely
                    formatted = []
                    now = datetime.now(timezone(timedelta(hours=-8)))  # PST
                    
                    for msg in messages:
                        # Parse timestamp with timezone
                        timestamp = datetime.fromtimestamp(
                            msg['timestamp'],
                            timezone(timedelta(hours=-8))  # PST
                        )
                        
                        # Calculate relative time
                        delta = now - timestamp
                        if delta.total_seconds() < 60:
                            relative_time = "just now"
                        elif delta.total_seconds() < 3600:
                            minutes = int(delta.total_seconds() / 60)
                            relative_time = f"{minutes}m ago"
                        elif delta.total_seconds() < 86400:
                            hours = int(delta.total_seconds() / 3600)
                            relative_time = f"{hours}h ago"
                        else:
                            days = int(delta.total_seconds() / 86400)
                            relative_time = f"{days}d ago"
                        
                        formatted.append(
                            f"[{relative_time}] "
                            f"{msg['author']['name']}: {msg['content']}"
                        )
                    
                    return "\n".join(formatted)
                    
        except Exception as e:
            logger.error(f"Error reading Discord channel: {str(e)}")
            return f"Failed to read Discord channel: {str(e)}"

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
            
        # Test Discord Reader
        print("\nTesting DiscordReaderTool:")
        try:
            reader = DiscordReaderTool(
                channel="general",
                lookback="1h"
            )
            result = await reader.call()
            print(f"Channel history:\n{result}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    asyncio.run(test_tools())