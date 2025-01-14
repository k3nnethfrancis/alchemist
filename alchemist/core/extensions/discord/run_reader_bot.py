"""Discord Reader Bot Service

This service must be running for the DiscordReaderTool to work.
Run this in a separate terminal before using the tool:

```bash
# Terminal 1: Start the reader bot service
python -m alchemist.core.extensions.discord.run_reader_bot

# Terminal 2: Run your chatbot with the DiscordReaderTool
python -m examples.discord.local_discord_reader
```
"""

import asyncio
import logging
from pathlib import Path
from aiohttp import web
import discord
from dotenv import load_dotenv
import os
from datetime import datetime, timezone, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure timezone (PST/PDT)
PST = timezone(timedelta(hours=-8))

class ReaderBot(discord.Client):
    """Discord bot for reading channel history."""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        super().__init__(intents=intents)
        self.ready = asyncio.Event()
        
    async def setup_hook(self):
        """Called when the client is done preparing data."""
        logger.info("Reader bot is setting up...")
        
    async def on_ready(self):
        """Called when the client is done preparing data after login."""
        logger.info(f"Reader bot logged in as {self.user.name} ({self.user.id})")
        
        # Log server and channel information
        logger.info("\n=== Available Servers and Channels ===")
        for guild in self.guilds:
            logger.info(f"\nServer: {guild.name} (ID: {guild.id})")
            logger.info("Channels:")
            
            # Group channels by category
            categories = {}
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    category_name = channel.category.name if channel.category else "No Category"
                    if category_name not in categories:
                        categories[category_name] = []
                    categories[category_name].append(channel)
            
            # Print channels by category
            for category, channels in categories.items():
                logger.info(f"\n  {category}:")
                for channel in channels:
                    logger.info(f"    #{channel.name} (ID: {channel.id})")
                    
        logger.info("\n=== Channel Quick Copy Format ===")
        logger.info("Copy this into your channel_map:")
        channel_map = {}
        for guild in self.guilds:
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    channel_map[channel.name] = str(channel.id)
        logger.info("channel_map = {")
        for name, id in channel_map.items():
            logger.info(f'    "{name}": "{id}",')
        logger.info("}")
        
        self.ready.set()
        
    async def fetch_channel_history(self, channel_id: int, days: float = 1.0):
        """Fetch message history from a channel."""
        await self.ready.wait()  # Ensure bot is ready
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                channel = await self.fetch_channel(channel_id)
                
            if not channel:
                raise ValueError(f"Could not find channel {channel_id}")
                
            logger.info(f"Fetching messages from #{channel.name} in server {channel.guild.name}")
            
            # Calculate time range
            now = datetime.now(PST)
            after = now - timedelta(days=days)
            logger.info(f"Fetching messages from {after} to {now}")
                
            messages = []
            async for message in channel.history(limit=100, after=after):
                # Include non-bot messages and bot messages with embeds
                if not message.author.bot or (message.author.bot and message.embeds):
                    # Convert to PST and ensure proper timezone info
                    timestamp_pst = message.created_at.astimezone(PST)
                    
                    # Handle regular message content
                    content = message.content
                    
                    # Handle embeds if present
                    if message.embeds:
                        for embed in message.embeds:
                            embed_content = []
                            
                            # Add embed title if present
                            if embed.title:
                                embed_content.append(f"**{embed.title}**")
                                
                            # Add embed description if present    
                            if embed.description:
                                embed_content.append(embed.description)
                                
                            # Add embed fields if present
                            for field in embed.fields:
                                embed_content.append(f"• {field.name}: {field.value}")
                                
                            # Combine embed content with original message
                            if embed_content:
                                content = content + "\n" + "\n".join(embed_content) if content else "\n".join(embed_content)
                    
                    messages.append({
                        "content": content,
                        "author": {
                            "name": message.author.name,
                            "id": str(message.author.id)
                        },
                        "timestamp": timestamp_pst.timestamp(),
                        "timezone": "PST",
                        "has_embeds": bool(message.embeds)
                    })
            
            logger.info(f"Found {len(messages)} messages")
            return messages
            
        except Exception as e:
            logger.error(f"Error fetching channel history: {str(e)}")
            raise

async def start_reader_bot():
    """Start the Discord reader bot and API server."""
    try:
        # Load environment variables
        load_dotenv()
        token = os.getenv("DISCORD_READER_TOKEN")
        if not token:
            raise ValueError("DISCORD_READER_TOKEN not set in environment")
            
        # Create bot instance
        bot = ReaderBot()
        
        # Setup API routes
        async def read_channel(request):
            """API endpoint to read channel history."""
            try:
                channel_id = int(request.query.get("channel_id"))
                days = float(request.query.get("days", 1.0))
                
                messages = await bot.fetch_channel_history(channel_id, days)
                return web.json_response(messages)
                
            except Exception as e:
                return web.Response(
                    status=400,
                    text=str(e)
                )
        
        # Create web app
        app = web.Application()
        app.router.add_get("/read_channel", read_channel)
        
        # Start bot and API server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", 5000)
        
        logger.info("Starting Discord reader bot service...")
        
        # Start both the bot and API server
        await asyncio.gather(
            bot.start(token),
            site.start()
        )
        
    except Exception as e:
        logger.error(f"Failed to start reader bot service: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(start_reader_bot())
    except KeyboardInterrupt:
        logger.info("Reader bot service stopped by user")
    except Exception as e:
        logger.error(f"Reader bot service failed: {str(e)}")
        raise 