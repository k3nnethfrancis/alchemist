"""Discord tool for reading channel history.

This module provides tools for:
- Reading channel history
- Extracting messages based on queries
- Supporting time-based filtering
- Handling embeds and attachments
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
from mirascope.core import BaseTool
from pydantic import BaseModel
import json
import re

logger = logging.getLogger(__name__)

class DiscordReaderTool(BaseTool):
    """Tool for reading Discord channel history.
    
    This tool connects to a local Discord bot service to read channel history.
    It supports both direct channel access and service mode, and can handle
    rich content like embeds and attachments.
    
    Example:
        ```python
        tool = DiscordReaderTool()
        await tool.configure(
            channels={"ai-news": "123456789"},
            categories={"ai": ["ai-news"]}
        )
        messages = await tool.call("Read #ai-news")
        ```
    """
    
    service_url: str = "http://localhost:5000"
    _channels: Dict[str, str] = {}
    _categories: Dict[str, List[str]] = {}
    
    def __init__(self, channels: dict = None):
        """Initialize the Discord reader tool.
        
        Args:
            channels: Optional dict mapping channel names to IDs
        """
        super().__init__()
        if channels:
            self.configure(channels)
        
    async def configure(self, channels: Dict[str, str], categories: Dict[str, List[str]]):
        """Configure the tool with channel mappings.
        
        Args:
            channels: Dict mapping channel names to IDs
            categories: Dict mapping category names to channel names
        """
        self._channels = channels
        self._categories = categories
        logger.info(f"Configured DiscordReaderTool with {len(channels)} channels")
        
    async def _format_message(self, msg: dict) -> str:
        """Format a single message with its embeds."""
        timestamp = msg.get('timestamp', '')
        author = msg.get('author', 'Unknown')
        content = msg.get('content', '')
        embeds = msg.get('embeds', [])
        
        formatted = f"[{timestamp}] {author}"
        if content:
            formatted += f": {content}"
        
        if embeds:
            formatted += "\nEmbeds:"
            for embed in embeds:
                if embed.get('title'):
                    formatted += f"\n  Title: {embed['title']}"
                if embed.get('description'):
                    desc = embed['description'].replace('\n', '\n    ')  # Indent description
                    formatted += f"\n  Description:\n    {desc}"
                if embed.get('fields'):
                    formatted += "\n  Fields:"
                    for field in embed['fields']:
                        formatted += f"\n    {field.get('name', '')}: {field.get('value', '')}"
        return formatted

    async def _call_service(self, channel_id: str) -> List[Dict[str, Any]]:
        """Call the Discord bot service to get messages from a channel.
        
        Args:
            channel_id: The ID of the channel to read messages from.
            
        Returns:
            List[Dict[str, Any]]: List of message objects containing:
                - id: Message ID
                - content: Message content
                - author: Author name
                - timestamp: Message timestamp
                - embeds: List of embeds with fields like title, description, fields, etc.
                - attachments: List of attachments
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.service_url}/history/{channel_id}?limit=100"
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                data = await response.json()
                messages = data.get("messages", [])
                
                # Process each message to ensure proper structure
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

    async def call(self, query: str) -> str:
        """Call the Discord reader tool with a query.
        
        Args:
            query: The query string in the format "Read #channel-name"
            
        Returns:
            str: JSON string containing messages from the channel
        """
        # Extract channel name from query
        match = re.search(r"Read #([a-zA-Z0-9-]+)", query)
        if not match:
            return json.dumps({"error": "Invalid query format. Use: Read #channel-name"})
        
        channel_name = match.group(1)
        
        # Get channel ID
        channel_id = self._channels.get(channel_name)
        if not channel_id:
            return json.dumps({"error": f"Channel '{channel_name}' not found"})
        
        # Get messages from service
        messages = await self._call_service(channel_id)
        
        return json.dumps({
            "messages": messages
        }) 