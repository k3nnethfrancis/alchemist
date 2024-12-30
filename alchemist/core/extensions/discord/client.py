"""
Discord Client Module

This module provides Discord integration for the ChatAgent.
"""

import logging
import discord
from typing import Optional, List, Dict, Any
import re
from datetime import datetime, timedelta
from pydantic import BaseModel
import asyncio
import os
from dotenv import load_dotenv

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import Graph, NodeState, NodeContext

logger = logging.getLogger(__name__)

class DiscordClient(discord.Client):
    """Discord client implementation for chat agents."""
    
    def __init__(self, agent: BaseAgent, intents: discord.Intents, token: str, workflow: Optional[Graph] = None):
        """Initialize the Discord client."""
        super().__init__(intents=intents)
        self.agent = agent
        self.token = token
        self.workflow = workflow  # Add workflow support

    async def setup_hook(self):
        """Called when the client is done preparing data."""
        logger.info("Bot is ready and setting up...")

    async def on_ready(self):
        """Called when the client is done preparing data after login."""
        logger.info(f"Logged in as {self.user.name} ({self.user.id})")

    async def on_message(self, message: discord.Message):
        """
        Handle incoming Discord messages.
        
        Args:
            message (discord.Message): The incoming message
        """
        try:
            # Ignore messages from bots (including self)
            if message.author.bot:
                logger.debug("Skipping message from bot")
                return

            # Check permissions first
            permissions = message.channel.permissions_for(message.guild.me)
            if not (permissions.read_messages and permissions.send_messages):
                logger.debug(f"Insufficient permissions in channel {message.channel.name}")
                return

            # Check if message is a mention
            is_mention = self.user.mentioned_in(message)
            
            # Clean the message content if it's a mention
            content = message.content
            if is_mention:
                content = content.replace(f"<@{self.user.id}>", "").strip()
                content = re.sub(r'<@!?[0-9]+>', '', content).strip()
            
            # Skip if content is empty after cleaning
            if not content:
                return
            
            # Convert to agent message format
            agent_message = {
                "content": content,
                "bot_id": str(self.user.id),
                "author": {
                    "id": str(message.author.id),
                    "name": message.author.name,
                    "bot": message.author.bot
                },
                "channel": {
                    "id": str(message.channel.id),
                    "name": message.channel.name
                },
                "mentions": [
                    {
                        "id": str(mention.id),
                        "name": mention.name,
                        "bot": mention.bot
                    }
                    for mention in message.mentions
                ],
                "timestamp": datetime.timestamp(message.created_at)
            }
            
            # Process through agent
            logger.info(f"Processing message: {content}")
            
            # If mentioned, handle directly
            if is_mention:
                if hasattr(self.agent, 'process_discord_message'):
                    response = await self.agent.process_discord_message(agent_message)
                else:
                    response = await self.agent._step(content)
                    
                if response:
                    logger.info(f"Sending direct response: {response}")
                    await message.channel.send(response)
            
            # If we have a workflow, process through that as well
            elif self.workflow:
                try:
                    state = NodeState(
                        context=NodeContext(
                            metadata={"recent_messages": [agent_message]}
                        )
                    )
                    
                    final_state = await self.workflow.run("main", state)
                    
                    if "engage" in final_state.results:
                        response = final_state.results["engage"]["response"]
                        logger.info(f"Sending workflow response: {response}")
                        await message.channel.send(response)
                        
                except Exception as e:
                    logger.error(f"Error in workflow processing: {str(e)}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing message: {error_msg}", exc_info=True)
            if message.channel.permissions_for(message.guild.me).send_messages:
                await message.channel.send(
                    f"I apologize, but I encountered an error while processing your message: {error_msg}. "
                    "Please try again or rephrase your request."
                )
            
    async def start(self):
        """Start the Discord client with the stored token."""
        await super().start(self.token)

    async def fetch_channel_history(self, channel_id: int, days: int = 2) -> List[Dict[str, Any]]:
        """Fetch message history from a specific channel with rich metadata."""
        try:
            logger.info(f"Attempting to fetch history from channel {channel_id}")
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Could not find channel {channel_id}")
                return []
            
            logger.info(f"Found channel: {channel.name}")
            messages = []
            start_date = datetime.now() - timedelta(days=days)
            
            logger.info(f"Fetching messages after {start_date}")
            async for message in channel.history(after=start_date, oldest_first=True):
                if not message.author.bot:
                    # Extract URLs and their metadata from content
                    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message.content)
                    
                    # Process embeds
                    embeds = []
                    for embed in message.embeds:
                        embed_data = {
                            "title": embed.title,
                            "description": embed.description,
                            "url": embed.url,
                            "timestamp": embed.timestamp.timestamp() if embed.timestamp else None,
                            "color": embed.color.value if embed.color else None,
                            "image": embed.image.url if embed.image else None,
                            "thumbnail": embed.thumbnail.url if embed.thumbnail else None,
                            "author": {
                                "name": embed.author.name if embed.author else None,
                                "url": embed.author.url if embed.author else None,
                                "icon_url": embed.author.icon_url if embed.author else None
                            } if embed.author else None,
                            "fields": [
                                {
                                    "name": field.name,
                                    "value": field.value,
                                    "inline": field.inline
                                } for field in embed.fields
                            ]
                        }
                        embeds.append(embed_data)
                    
                    # Process attachments
                    attachments = [
                        {
                            "id": str(attachment.id),
                            "filename": attachment.filename,
                            "url": attachment.url,
                            "content_type": attachment.content_type,
                            "size": attachment.size,
                            "height": attachment.height,
                            "width": attachment.width,
                        } for attachment in message.attachments
                    ]
                    
                    messages.append({
                        "content": message.content,
                        "clean_content": message.clean_content,  # Content with mentions/channels replaced
                        "author": {
                            "id": str(message.author.id),
                            "name": message.author.name,
                            "display_name": message.author.display_name,
                            "bot": message.author.bot,
                            "avatar_url": str(message.author.avatar.url) if message.author.avatar else None
                        },
                        "channel": {
                            "id": str(message.channel.id),
                            "name": message.channel.name
                        },
                        "timestamp": datetime.timestamp(message.created_at),
                        "edited_timestamp": datetime.timestamp(message.edited_at) if message.edited_at else None,
                        "urls": urls,
                        "embeds": embeds,
                        "attachments": attachments,
                        "reference": {
                            "message_id": str(message.reference.message_id),
                            "channel_id": str(message.reference.channel_id),
                            "guild_id": str(message.reference.guild_id)
                        } if message.reference else None,
                        "flags": message.flags.value,
                        "type": str(message.type)
                    })
            
            logger.info(f"Found {len(messages)} messages with rich metadata")
            return messages
            
        except Exception as e:
            logger.error(f"Error fetching channel history: {str(e)}", exc_info=True)
            return []

    @staticmethod
    async def test_fetch_history():
        """Test function to verify channel history fetching."""
        try:
            # Load environment
            from dotenv import load_dotenv
            load_dotenv()
            
            # Use reader tokens for testing
            token = os.getenv("DISCORD_READER_TOKEN")
            channel_id = os.getenv("DISCORD_READER_CHANNEL_ID")
            days = 7
            
            # Validate environment variables
            if not token:
                raise ValueError("DISCORD_READER_TOKEN not found in environment")
            if not channel_id:
                raise ValueError("DISCORD_READER_CHANNEL_ID not found in environment")
            
            try:
                channel_id = int(channel_id)
            except ValueError:
                raise ValueError(f"Invalid DISCORD_READER_CHANNEL_ID: {channel_id}. Must be a valid integer.")
            
            logger.info(f"Using token: {token[:8]}... for channel: {channel_id}")
            
            # Setup client
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True
            
            client = DiscordClient(
                agent=None,  # Not needed for testing
                intents=intents,
                token=token
            )
            
            logger.info(f"Starting test fetch from channel {channel_id} for last {days} days...")
            
            # Create task for client start
            asyncio.create_task(client.start())
            
            # Wait for ready
            await asyncio.sleep(2)
            
            try:
                # Fetch messages
                messages = await client.fetch_channel_history(channel_id, days=days)
                
                # Print detailed results
                print("\n=== Test Results ===")
                print(f"Total messages found: {len(messages)}")
                
                for idx, msg in enumerate(messages, 1):
                    print(f"\nMessage {idx}:")
                    print(f"Author: {msg['author']['name']}")
                    print(f"Content: {msg['content'][:100]}...")  # First 100 chars
                    print(f"Timestamp: {datetime.fromtimestamp(msg['timestamp'])}")
                    
                    if msg['embeds']:
                        print(f"Embeds ({len(msg['embeds'])}):")
                        for embed in msg['embeds']:
                            print(f"  - Title: {embed['title']}")
                            print(f"  - Description: {embed['description'][:100]}..." if embed['description'] else "  - No description")
                            print(f"  - URL: {embed['url']}")
                            if embed['image']:
                                print(f"  - Image: {embed['image']}")
                            
                    if msg['urls']:
                        print(f"URLs: {msg['urls']}")
                        
                    if msg['attachments']:
                        print(f"Attachments ({len(msg['attachments'])}):")
                        for att in msg['attachments']:
                            print(f"  - {att['filename']} ({att['content_type']})")
                    
                    print("-" * 50)
            
            finally:
                # Ensure we always cleanup
                logger.info("Cleaning up client connection...")
                if not client.is_closed():
                    await client.close()
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}", exc_info=True)
            print("\nMake sure you have a .env file with the following variables:")
            print("DISCORD_READER_TOKEN=xxx")
            print("DISCORD_READER_CHANNEL_ID=xxx")
            raise

if __name__ == "__main__":
    # Run test
    logging.basicConfig(level=logging.INFO)
    asyncio.run(DiscordClient.test_fetch_history())