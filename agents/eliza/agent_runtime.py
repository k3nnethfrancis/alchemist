"""
AgentRuntime Module - Core runtime for the Eliza Discord bot.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import discord
import asyncio
from mirascope.core import Messages, prompt_template, openai
from eliza.memory.message_history import MessageHistory

logger = logging.getLogger(__name__)

class AgentRuntime:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.message_history = None
        self.agent_profile = None
        self.last_analysis = {}  # Store datetime objects
        self.bot_user_id = None

    async def initialize(self):
        """Initialize runtime components."""
        self.message_history = MessageHistory()
        await self.message_history.initialize()
        
    def set_agent_profile(self, profile: Dict[str, Any]):
        """Set the agent's personality profile."""
        self.agent_profile = profile

    async def periodic_analysis(self, client: discord.Client):
        """
        Periodically analyze channels for conversation opportunities.
        
        Args:
            client: Discord client instance
        """
        await client.wait_until_ready()
        while not client.is_closed():
            try:
                now = datetime.utcnow()
                
                for channel in client.get_all_channels():
                    if not isinstance(channel, discord.TextChannel):
                        continue
                        
                    channel_id = str(channel.id)
                    last_check = self.last_analysis.get(channel_id, datetime.utcnow() - timedelta(minutes=10))
                    
                    if (now - last_check) < timedelta(seconds=30):
                        continue
                        
                    self.last_analysis[channel_id] = now
                    
                    # Get recent messages
                    async for message in channel.history(limit=10):
                        if message.author.bot:
                            continue
                            
                        messages = await self.message_history._get_channel_messages(channel_id)
                        context = "\n".join(messages)
                        
                        should_respond = await self.should_respond(
                            name=self.agent_profile["name"],
                            bio=self.agent_profile["bio"],
                            context=context,
                            message=message.content
                        )
                        
                        if "RESPOND" in should_respond.content.upper():
                            response = await self.generate_response(
                                name=self.agent_profile["name"],
                                bio=self.agent_profile["bio"],
                                style_guidelines="\n".join(self.agent_profile["style"]["chat"]),
                                context=context,
                                message=message.content
                            )
                            
                            await channel.send(response.content)
                            await self.message_history.add_message(
                                channel_id,
                                f"{message.author.name}: {message.content}"
                            )
                            await self.message_history.add_message(
                                channel_id,
                                f"{self.agent_profile['name']}: {response.content}"
                            )
                            break
                            
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in periodic analysis: {str(e)}")
                await asyncio.sleep(30)

    @openai.call("gpt-4o-mini")
    @prompt_template("""
    You are {name}, {bio}

    Recent conversation:
    {context}

    Current message: {message}

    Should you respond? Consider:
    1. Is the message directed at you?
    2. Is the conversation relevant to your interests?
    3. Would your response add value?
    4. Has enough time passed since your last message?

    Respond with only: RESPOND, IGNORE, or CONTINUE
    """)
    async def should_respond(self, name: str, bio: str, context: str, message: str) -> str:
        """Determine if the agent should respond."""
        return {}

    @openai.call("gpt-4o-mini")
    @prompt_template("""
    You are {name}, {bio}

    Your style guidelines:
    {style_guidelines}

    Recent conversation:
    {context}

    Current message: {message}

    Respond in character:
    """)
    async def generate_response(self, name: str, bio: str, style_guidelines: str, 
                              context: str, message: str) -> str:
        """Generate a response in character."""
        return {}

    async def handle_message(self, message: discord.Message) -> Optional[str]:
        """Process a Discord message and generate a response if appropriate."""
        try:
            if message.author.bot:
                return None

            context = await self.message_history.get_history(str(message.channel.id))
            
            should_respond = await self.should_respond(
                name=self.agent_profile["name"],
                bio=self.agent_profile["bio"],
                context=context,
                message=message.content
            )
            
            if "RESPOND" in should_respond.content.upper():
                response = await self.generate_response(
                    name=self.agent_profile["name"],
                    bio=self.agent_profile["bio"],
                    style_guidelines=self.agent_profile["style_guidelines"],
                    context=context,
                    message=message.content
                )
                
                await self.message_history.add_message(
                    str(message.channel.id),
                    f"{message.author.name}: {message.content}"
                )
                await self.message_history.add_message(
                    str(message.channel.id),
                    f"{self.agent_profile['name']}: {response.content}"
                )
                
                return response.content
                
            return None
            
        except Exception as e:
            self.logger.error({
                "event": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return "I encountered an error processing your message."

    async def shutdown(self):
        """Cleanup resources before shutdown."""
        logger.info("Shutting down AgentRuntime...")
        if self.message_history:
            await self.message_history.cleanup()