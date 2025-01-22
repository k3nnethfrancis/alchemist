"""Discord toolkit for reading channel history.

This module provides tools for reading message history from Discord channels
via a local bot service. Features:
- Channel history retrieval with name-based lookup
- Time-based filtering
- Rich content support (embeds, attachments)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
from mirascope.core import BaseTool, BaseToolKit, toolkit_tool
from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)

class DiscordTools(BaseToolKit):
    """A toolkit for reading Discord channel history.
    
    This toolkit provides tools for interacting with Discord channels and
    automatically handles channel name to ID mapping.
    
    Attributes:
        channels: Mapping of channel names to IDs
        categories: Mapping of category names to channel lists
        service_url: URL of the local Discord bot service
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
        try:
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
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error reading Discord channel: {str(e)}")
            raise Exception(f"Failed to get messages: {str(e)}")
        except ValueError as e:
            logger.error(f"Error reading Discord channel: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error reading Discord channel: {str(e)}")
            raise Exception(f"Failed to get messages: {str(e)}")

    def get_system_prompt(self) -> str:
        """Get the system prompt extension for Discord tools.
        
        Returns:
            str: System prompt extension with available channels and usage instructions
        """
        prompt = "\nAvailable Discord channels:\n"
        for category, channel_list in self.categories.items():
            prompt += f"\n{category}:\n"
            for channel in channel_list:
                prompt += f"  #{channel}\n"
                
        prompt += "\nWhen using the Discord reader tool:"
        prompt += "\n1. Use the channel name (with or without #)"
        prompt += "\n2. Optionally specify a time filter"
        prompt += "\n3. Messages will be returned newest first"
        return prompt 