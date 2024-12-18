"""
Tool to read the latest messages from a Discord channel.
"""

from typing import Optional
from mirascope.core import BaseTool
import discord
import asyncio

class ReadDiscordMessages(BaseTool):
    """
    Reads recent messages from a specified Discord channel.
    """
    name = "read_discord_messages"
    description = "Reads the latest messages from a Discord channel."

    def __init__(self, client: discord.Client):
        self.client = client

    async def _run(self, channel_id: int, limit: int = 10) -> str:
        channel = self.client.get_channel(channel_id)
        if not channel:
            return f"Channel with ID {channel_id} not found."

        messages = await channel.history(limit=limit).flatten()
        messages_text = "\n".join(
            [f"{msg.author.name}: {msg.content}" for msg in reversed(messages)]
        )
        return messages_text

    async def __call__(self, channel_id: int, limit: int = 10) -> str:
        return await self._run(channel_id, limit)