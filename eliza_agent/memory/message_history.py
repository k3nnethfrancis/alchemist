"""
Message history memory management for the Eliza Discord agent.
Maintains a rolling window of recent messages per channel with SQLite persistence.
"""

import logging
import aiosqlite
from typing import Dict, List, Any
from datetime import datetime
from .base import BaseMemory

logger = logging.getLogger(__name__)

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
)
"""

CREATE_CHANNELS_TABLE = """
CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY,
    last_accessed REAL NOT NULL
)
"""

class MessageHistory(BaseMemory):
    """
    Manages message history for Discord channels with SQLite persistence.
    
    Attributes:
        max_messages (int): Maximum number of messages to store per channel
        db_path (str): Path to SQLite database
        db (aiosqlite.Connection): Database connection
    """
    
    def __init__(self, max_messages: int = 50, db_path: str = "memory.db"):
        """
        Initialize the message history manager.
        
        Args:
            max_messages (int, optional): Maximum messages per channel. Defaults to 50.
            db_path (str, optional): Path to SQLite database. Defaults to "memory.db".
        """
        self.max_messages = max_messages
        self.db_path = db_path
        self.db = None
        self._channel_cache: Dict[str, List[str]] = {}
        
    async def initialize(self):
        """Initialize the database connection and create tables if they don't exist."""
        logger.info(f"Initializing MessageHistory with database: {self.db_path}")
        self.db = await aiosqlite.connect(self.db_path)
        
        # Create tables
        await self.db.execute(CREATE_CHANNELS_TABLE)
        await self.db.execute(CREATE_MESSAGES_TABLE)
        await self.db.commit()
        
        # Create indices for better performance
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_channel_messages ON messages(channel_id, timestamp)"
        )
        await self.db.commit()
        
    async def cleanup(self):
        """Close the database connection."""
        if self.db:
            await self.db.close()
            self.db = None
        
    async def get_context(self) -> Dict[str, Any]:
        """
        Get the current message history context.
        
        Returns:
            Dict[str, Any]: Current message histories by channel
        """
        if not self.db:
            await self.initialize()
            
        histories = {}
        async with self.db.execute(
            """
            SELECT DISTINCT channel_id FROM channels 
            ORDER BY last_accessed DESC 
            LIMIT 10
            """
        ) as cursor:
            async for (channel_id,) in cursor:
                histories[channel_id] = await self._get_channel_messages(channel_id)
                
        return {"message_histories": histories}
        
    async def add_message(self, channel_id: str, message: str):
        """
        Add a new message to the history.
        
        Args:
            channel_id (str): The Discord channel ID
            message (str): The message to store
        """
        if not self.db:
            await self.initialize()
            
        current_time = datetime.utcnow().timestamp()
        
        # Update or insert channel
        await self.db.execute(
            """
            INSERT INTO channels (channel_id, last_accessed) 
            VALUES (?, ?) 
            ON CONFLICT(channel_id) DO UPDATE SET last_accessed = ?
            """,
            (channel_id, current_time, current_time)
        )
        
        # Add new message
        await self.db.execute(
            "INSERT INTO messages (channel_id, content, timestamp) VALUES (?, ?, ?)",
            (channel_id, message, current_time)
        )
        
        # Cleanup old messages
        await self._cleanup_channel_messages(channel_id)
        
        # Update cache
        if channel_id in self._channel_cache:
            self._channel_cache[channel_id].append(message)
            if len(self._channel_cache[channel_id]) > self.max_messages:
                self._channel_cache[channel_id].pop(0)
                
        await self.db.commit()
        
    async def _get_channel_messages(self, channel_id: str) -> List[str]:
        """
        Get messages for a specific channel.
        
        Args:
            channel_id (str): The Discord channel ID
            
        Returns:
            List[str]: List of messages for the channel
        """
        if channel_id in self._channel_cache:
            return self._channel_cache[channel_id]
            
        messages = []
        async with self.db.execute(
            """
            SELECT content FROM messages 
            WHERE channel_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """,
            (channel_id, self.max_messages)
        ) as cursor:
            async for (content,) in cursor:
                messages.append(content)
                
        messages.reverse()  # Put in chronological order
        self._channel_cache[channel_id] = messages
        return messages
        
    async def _cleanup_channel_messages(self, channel_id: str):
        """
        Remove old messages to maintain the maximum message limit.
        
        Args:
            channel_id (str): The Discord channel ID
        """
        # Get the timestamp of the oldest message we want to keep
        async with self.db.execute(
            """
            SELECT timestamp FROM messages 
            WHERE channel_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1 OFFSET ?
            """,
            (channel_id, self.max_messages)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                cutoff_time = row[0]
                # Delete older messages
                await self.db.execute(
                    "DELETE FROM messages WHERE channel_id = ? AND timestamp < ?",
                    (channel_id, cutoff_time)
                )