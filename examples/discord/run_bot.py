"""Discord Reader Bot Service.

This script runs a Discord bot service that provides:
1. Channel information via HTTP endpoint
2. Message history access via HTTP endpoint

Before running:
1. Make sure you have set DISCORD_READER_TOKEN in your .env file
2. Run this in a separate terminal before starting the local Discord reader
"""

import asyncio
import logging
import os
from typing import Dict, List
from aiohttp import web
from dotenv import load_dotenv

from alchemist.core.extensions.discord.runtime import DiscordRuntime, DiscordRuntimeConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiscordBotService:
    """Discord bot service for channel access."""
    
    def __init__(self):
        """Initialize the bot service."""
        self.runtime = None
        self.ready = asyncio.Event()
        self.channels: Dict[str, str] = {}  # name -> id
        self.categories: Dict[str, List[str]] = {}  # category -> [channel names]
        
    async def start(self):
        """Start the Discord runtime and HTTP server."""
        try:
            # Load environment variables
            load_dotenv()
            token = os.getenv("DISCORD_READER_TOKEN")
            if not token:
                raise ValueError(
                    "DISCORD_READER_TOKEN not found in environment. "
                    "Please set it in your .env file."
                )
                
            logger.info("Starting Discord reader bot...")
            
            # Configure Discord runtime
            runtime_config = DiscordRuntimeConfig(
                bot_token=token,
                channel_ids=["*"]  # Allow access to all channels
            )
            
            self.runtime = DiscordRuntime(config=runtime_config)
            
            # Add message handler for logging
            self.runtime.add_message_handler(self._handle_message)
            
            # Start Discord runtime
            await self.runtime.start()
            logger.info("Discord runtime started successfully")
            
            # Update channel information
            await self._update_channels()
            logger.info(f"Found {len(self.channels)} channels in {len(self.categories)} categories")
            self.ready.set()
            
            # Start HTTP server
            logger.info("Starting HTTP server...")
            app = web.Application()
            app.router.add_get('/channels', self.handle_channels)
            app.router.add_get('/history/{channel_id}', self.handle_history)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 5000)
            await site.start()
            
            logger.info("Discord reader bot service running on http://localhost:5000")
            logger.info("Available endpoints:")
            logger.info("  - GET /channels")
            logger.info("  - GET /history/{channel_id}?limit={limit}")
            
            # Keep the service running
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Shutting down HTTP server...")
                await runner.cleanup()
                logger.info("HTTP server stopped")
                
        except Exception as e:
            logger.error(f"Error starting service: {str(e)}")
            raise
    
    async def _handle_message(self, message):
        """Log incoming messages with full details."""
        embed_info = []
        for embed in message.embeds:
            embed_data = {
                "title": embed.title,
                "description": embed.description,
                "url": embed.url,
                "fields": [{"name": f.name, "value": f.value} for f in embed.fields],
                "footer": {"text": embed.footer.text} if embed.footer else None,
                "author": {"name": embed.author.name} if embed.author else None,
                "color": embed.color if embed.color else None,
                "thumbnail": {"url": embed.thumbnail.url} if embed.thumbnail else None,
                "image": {"url": embed.image.url} if embed.image else None
            }
            embed_info.append(embed_data)
            
        logger.info(
            f"[Discord] Message from {message.author}:\n"
            f"Content: {message.content}\n"
            f"Embeds: {embed_info if embed_info else 'None'}"
        )
    
    async def _update_channels(self):
        """Update channel mappings from Discord."""
        logger.info("Updating channel mappings...")
        channels = await self.runtime.get_channels()
        self.channels = channels["channels"]
        self.categories = channels["categories"]
        
        # Log available channels
        for category, channel_list in self.categories.items():
            logger.info(f"\nCategory: {category}")
            for channel in channel_list:
                logger.info(f"  - #{channel}")
    
    async def handle_channels(self, request):
        """Handle GET /channels request."""
        logger.info("Handling /channels request")
        return web.json_response({
            "channels": self.channels,
            "categories": self.categories
        })
    
    async def handle_history(self, request):
        """Handle GET /history/{channel_id} request."""
        channel_id = request.match_info['channel_id']
        limit = int(request.query.get('limit', 100))
        
        logger.info(f"Handling /history request for channel {channel_id} (limit: {limit})")
        
        try:
            messages = await self.runtime.get_message_history(channel_id, limit)
            logger.info(f"Found {len(messages)} messages")
            return web.json_response({"messages": messages})
        except ValueError as e:
            logger.error(f"Channel not found: {str(e)}")
            raise web.HTTPNotFound(text=str(e))

async def main():
    """Run the Discord bot service."""
    try:
        service = DiscordBotService()
        await service.start()
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDiscord reader bot service terminated by user.") 