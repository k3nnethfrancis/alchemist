"""Test implementation of a basic Mirascope toolkit.

This module demonstrates the basic usage of Mirascope's toolkit pattern
with async operations for Discord integration.
"""

from typing import Dict, List, Literal, Optional, Any
from datetime import datetime, timedelta, timezone
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
from openpipe import OpenAI as OpenPipeClient

"""
class BookTools(BaseToolKit):
    # Commented out book recommendation example
    __namespace__ = "book_tools"
    
    reading_level: Literal["beginner", "advanced"] = Field(
        description="The user's reading level"
    )

    @toolkit_tool
    def suggest_author(self, author: str) -> str:
        return f"I would suggest you read some books by {author}"
"""

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
        """Read messages from a Discord channel.

        Available channels: {self.channels}
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

"""
@openai.call("gpt-4o-mini", client=OpenPipeClient())
def recommend_author(
    genre: str, 
    reading_level: Literal["beginner", "advanced"]
) -> BaseDynamicConfig:
    # Commented out book recommendation example
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
"""

@openai.call("gpt-4o-mini", client=OpenPipeClient())
def test_discord_toolkit(channel_name: str = "ai-news") -> BaseDynamicConfig:
    """Test the Discord toolkit pattern by requesting recent AI news."""
    # Load channel data from config file
    print("\nLoading channel data from config...")
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "channels.json"
    with open(config_path) as f:
        data = json.load(f)
        channels = data["channels"]
        categories = data["categories"]
        print(f"Found {len(channels)} channels in {len(categories)} categories")
    
    toolkit = DiscordTools(
        channels=channels,
        categories=categories
    )
    
    return {
        "tools": toolkit.create_tools(),
        "messages": [
            BaseMessageParam(
                role="system",
                content="You are a helpful assistant that can read Discord channels and summarize their contents."
            ),
            BaseMessageParam(
                role="user",
                content=f"Tell me about the recent messages in the #{channel_name} channel. Summarize the key updates."
            )
        ],
    }

async def main():
    """Run the Discord toolkit example."""
    print("\nTesting Discord AI News Summary...")
    
    response = test_discord_toolkit()
    if tool := response.tool:
        try:
            messages = await tool.call()
            print(f"\nRetrieved {len(messages)} messages from #ai-news")
            if messages:
                print("\nLatest messages:")
                for msg in messages[:5]:  # Show last 5 messages
                    print(f"\nFrom {msg['author']} at {msg['timestamp']}:")
                    print(f"Content: {msg['content']}")
                    if msg['embeds']:
                        print(f"Embeds: {len(msg['embeds'])}")
                        for embed in msg['embeds']:
                            if embed['title']:
                                print(f"- {embed['title']}")
                            if embed['description']:
                                print(f"  {embed['description'][:200]}...")
            else:
                print("No messages found in the last 2 hours")
        except Exception as e:
            print(f"Error reading ai-news: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 