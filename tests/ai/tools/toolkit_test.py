"""Test implementation of a basic Mirascope toolkit.

This module demonstrates the basic usage of Mirascope's toolkit pattern
following their documentation example exactly.
"""

from typing import Dict, List, Literal, Optional, Any
from datetime import datetime
import aiohttp
from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    BaseToolKit,
    openai,
    toolkit_tool,
)
from pydantic import Field
import json
from pathlib import Path


class BookTools(BaseToolKit):
    """A toolkit for book recommendations.
    
    This follows the exact implementation from Mirascope's documentation
    to verify our understanding of the toolkit pattern.
    
    Attributes:
        reading_level: The user's reading level (beginner or advanced)
    """
    
    __namespace__ = "book_tools"
    
    reading_level: Literal["beginner", "advanced"] = Field(
        description="The user's reading level"
    )

    @toolkit_tool
    def suggest_author(self, author: str) -> str:
        """Suggests an author for the user to read based on their reading level.

        Reading level: {self.reading_level}
        """
        return f"I would suggest you read some books by {author}"


class DiscordTools(BaseToolKit):
    """A toolkit for reading Discord channel history.
    
    Attributes:
        channels: Mapping of channel names to IDs
        categories: Mapping of category names to channel lists
    """
    
    __namespace__ = "discord_tools"
    
    channels: Dict[str, str] = Field(
        description="Mapping of channel names to IDs"
    )
    categories: Dict[str, List[str]] = Field(
        description="Mapping of category names to channel lists"
    )
    service_url: str = Field(
        default="http://localhost:5000",
        description="URL of the local Discord bot service"
    )

    @toolkit_tool
    async def read_channel(
        self,
        channel_name: str,
        after: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Read message history from a Discord channel.

        Available channels: {', '.join(self.channels.keys())}
        """
        # Strip # if present and look up channel ID
        clean_name = channel_name.lstrip('#')
        channel_id = self.channels.get(clean_name)
        if not channel_id:
            raise ValueError(f"Channel '{channel_name}' not found")
            
        url = f"{self.service_url}/history/{channel_id}?limit={limit}"
        if after:
            url += f"&after={after.isoformat()}"
            
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get messages: {response.status}")
                    
                data = await response.json()
                messages = data.get("messages", [])
                
                # Process messages to ensure consistent structure
                processed_messages = []
                for msg in messages:
                    processed_msg = {
                        "id": msg.get("id"),
                        "content": msg.get("content", ""),
                        "author": msg.get("author") if isinstance(msg.get("author"), str) else msg.get("author", {}).get("name", "Unknown"),
                        "timestamp": msg.get("timestamp"),
                        "embeds": [{
                            "title": embed.get("title"),
                            "description": embed.get("description"),
                            "fields": embed.get("fields", []),
                            "color": embed.get("color"),
                            "footer": embed.get("footer"),
                            "thumbnail": embed.get("thumbnail"),
                            "url": embed.get("url")
                        } for embed in msg.get("embeds", [])],
                        "attachments": msg.get("attachments", [])
                    }
                    processed_messages.append(processed_msg)
                
                return processed_messages


@openai.call("gpt-4o-mini")
def recommend_author(
    genre: str, 
    reading_level: Literal["beginner", "advanced"]
) -> BaseDynamicConfig:
    """Recommend an author based on genre and reading level.
    
    Args:
        genre: The desired book genre
        reading_level: The user's reading level
        
    Returns:
        BaseDynamicConfig: Configuration with tools and messages
    """
    toolkit = BookTools(reading_level=reading_level)
    return {
        "tools": toolkit.create_tools(),
        "messages": [
            BaseMessageParam(
                role="user",
                content=f"What {genre} author should I read?"
            )
        ],
    }


@openai.call("gpt-4o-mini")
async def read_discord_history(
    channel_name: str,
    time_filter: Optional[datetime] = None,
    channels: Dict[str, str] = {},
    categories: Dict[str, List[str]] = {}
) -> BaseDynamicConfig:
    """Read message history from a Discord channel.
    
    Args:
        channel_name: Name of the channel to read from (without #)
        time_filter: Optional timestamp to filter messages after
        channels: Mapping of channel names to IDs
        categories: Mapping of category names to channel lists
        
    Returns:
        BaseDynamicConfig: Configuration with tools and messages
    """
    toolkit = DiscordTools(
        channels=channels,
        categories=categories
    )
    
    return {
        "tools": toolkit.create_tools(),
        "messages": [
            BaseMessageParam(
                role="user",
                content=f"Read messages from #{channel_name}"
                + (f" after {time_filter.isoformat()}" if time_filter else "")
            )
        ],
    }


async def test_discord():
    """Test the Discord toolkit implementation."""
    # Load channel data from config file
    print("\nLoading channel data from config...")
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "channels.json"
    with open(config_path) as f:
        data = json.load(f)
        channels = data["channels"]
        categories = data["categories"]
        print(f"Found {len(channels)} channels in {len(categories)} categories")
        print("\nAvailable channels:")
        for name, id in channels.items():
            print(f"  #{name}: {id}")
    
    # Test reading channel history
    print("\nTesting with real Discord service...")
    
    # First try ai-news channel
    channel_name = "ai-news"
    channel_id = channels.get(channel_name)
    print(f"\nTrying to read #{channel_name} (ID: {channel_id})")
    
    response = await read_discord_history(
        channel_name=channel_name,
        channels=channels,
        categories=categories
    )
    if tool := response.tool:
        try:
            messages = await tool.call()
            print(f"Retrieved {len(messages)} messages from #{channel_name}")
            if messages:
                print("Latest message:", messages[0]["content"][:100] + "...")
        except Exception as e:
            print(f"Error reading {channel_name}: {str(e)}")
            
    # Then try agent-sandbox
    channel_name = "agent-sandbox"
    channel_id = channels.get(channel_name)
    print(f"\nTrying to read #{channel_name} (ID: {channel_id})")
    
    response = await read_discord_history(
        channel_name=channel_name,
        channels=channels,
        categories=categories
    )
    if tool := response.tool:
        try:
            messages = await tool.call()
            print(f"Retrieved {len(messages)} messages from #{channel_name}")
            if messages:
                print("Latest message:", messages[0]["content"][:100] + "...")
        except Exception as e:
            print(f"Error reading {channel_name}: {str(e)}")


if __name__ == "__main__":
    # Test book recommendations
    print("\nTesting Book Recommendations:")
    response = recommend_author("fantasy", "beginner")
    if tool := response.tool:
        print("Beginner recommendation:", tool.call())
    
    response = recommend_author("fantasy", "advanced")
    if tool := response.tool:
        print("Advanced recommendation:", tool.call())
    
    # Test Discord toolkit
    print("\nTesting Discord Toolkit:")
    import asyncio
    asyncio.run(test_discord()) 