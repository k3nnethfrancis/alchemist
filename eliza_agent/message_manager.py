"""
Message Manager Module

Determines if the bot should respond based on cooldowns and mentions, and manages cooldown persistence.

Classes:
    MessageManager: Handles cooldown logic and response decision-making.
"""

import logging
import aiosqlite
from datetime import datetime
import discord
from discord import Message
from core.models.types import RuntimeConfig

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 30
DATABASE_FILE = 'cooldowns.db'


class MessageManager:
    """
    Manages message handling for the agent.

    Attributes:
        runtime_config (RuntimeConfig): The runtime configuration.
        db (aiosqlite.Connection): Asynchronous connection to the SQLite database.
    """

    def __init__(self, runtime_config: RuntimeConfig):
        """
        Initializes the MessageManager.

        Args:
            runtime_config (RuntimeConfig): The runtime configuration.
        """
        self.runtime_config = runtime_config
        self.bot_user_id = None  # Will be set later by DiscordClient
        self.db_path = DATABASE_FILE
        self.cooldown_seconds = COOLDOWN_SECONDS

    async def initialize(self):
        """
        Initializes the SQLite database and creates the cooldowns table if it doesn't exist.
        """
        self.db = await aiosqlite.connect(DATABASE_FILE)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS cooldowns (
                channel_id TEXT PRIMARY KEY,
                last_response_time REAL
            )
        """)
        await self.db.commit()
        logger.info("Initialized SQLite database for cooldowns.")

    async def close(self):
        """
        Closes the SQLite database connection.
        """
        if self.db:
            await self.db.close()
            logger.info("Closed SQLite database connection for cooldowns.")

    async def should_respond(self, message: Message) -> bool:
        """
        Determines if the bot should respond to a message.
        """
        if not self.bot_user_id:
            logger.warning("Bot user ID is not set in MessageManager.")
            return False

        # Always respond to mentions
        if self.bot_user_id in [mention.id for mention in message.mentions]:
            return True

        # Check cooldown for non-mentions
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT last_response_time FROM cooldowns WHERE channel_id = ?",
                (str(message.channel.id),)
            )
            result = await cursor.fetchone()
            
            current_time = datetime.utcnow().timestamp()
            if result:
                last_response_time = result[0]
                if current_time - last_response_time < self.cooldown_seconds:
                    return False
                    
            await db.execute(
                "INSERT OR REPLACE INTO cooldowns (channel_id, last_response_time) VALUES (?, ?)",
                (str(message.channel.id), current_time)
            )
            await db.commit()
            
        return True

    async def get_last_response_time(self, channel_id: int) -> float:
        """
        Retrieves the last response time for a given channel.

        Args:
            channel_id (int): The ID of the Discord channel.

        Returns:
            float: The timestamp of the last response, or None if not found.
        """
        async with self.db.execute("SELECT last_response_time FROM cooldowns WHERE channel_id = ?", (str(channel_id),)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            return None

    async def update_cooldown(self, channel_id: int):
        """
        Updates the last response time for a given channel to the current time.

        Args:
            channel_id (int): The ID of the Discord channel.
        """
        current_time = datetime.utcnow().timestamp()
        await self.db.execute("""
            INSERT INTO cooldowns (channel_id, last_response_time)
            VALUES (?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET last_response_time=excluded.last_response_time
        """, (str(channel_id), current_time))
        await self.db.commit()
        logger.debug(f"Updated cooldown for channel {channel_id} at {current_time}.")