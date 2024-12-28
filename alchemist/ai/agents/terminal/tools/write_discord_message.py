"""
Tool to write messages to a Discord channel with cooldown management.
"""

from mirascope.core import BaseTool
import discord
import asyncio
import time

class WriteDiscordMessage(BaseTool):
    """
    Sends a message to a specified Discord channel with a cooldown.
    """
    name = "write_discord_message"
    description = "Writes a message to a Discord channel."

    cooldown_time = 10  # Cooldown in seconds
    last_written_time = 0

    def __init__(self, client: discord.Client):
        self.client = client

    async def _run(self, channel_id: int, message: str) -> str:
        current_time = time.time()
        if current_time - self.last_written_time < self.cooldown_time:
            remaining = self.cooldown_time - (current_time - self.last_written_time)
            return f"Cooldown active. Please wait {remaining:.1f} seconds."

        channel = self.client.get_channel(channel_id)
        if not channel:
            return f"Channel with ID {channel_id} not found."

        await channel.send(message)
        self.last_written_time = current_time
        return "Message sent successfully."

    async def __call__(self, channel_id: int, message: str) -> str:
        return await self._run(channel_id, message)