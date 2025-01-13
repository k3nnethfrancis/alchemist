"""Runtime for local chat with Discord reading capabilities."""

from typing import Optional
import discord
from pathlib import Path

from alchemist.ai.base.runtime import BaseChatRuntime, RuntimeConfig
from alchemist.core.extensions.discord.client import DiscordClient

class DiscordReaderRuntime(BaseChatRuntime):
    """Runtime for local chat with Discord reading capabilities."""
    
    def __init__(self, token: str, config: RuntimeConfig):
        super().__init__(config)
        self.token = token
        self.discord_client = None
        
    async def setup_discord(self):
        """Setup Discord client for reading."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        
        self.discord_client = DiscordClient(
            agent=self.agent,
            intents=intents,
            token=self.token
        )
        # Start client in background
        await self.discord_client.start()
        
    async def read_channel(self, channel_id: int, days: int = 2) -> str:
        """Read messages from a Discord channel."""
        if not self.discord_client:
            await self.setup_discord()
            
        messages = await self.discord_client.fetch_channel_history(
            channel_id=channel_id,
            days=days
        )
        return messages
    
    async def start(self) -> None:
        """Start a local chat session with Discord reading capability."""
        self._start_session("discord_reader")
        print("\nStarting enhanced chat session. Type 'exit' or 'quit' to stop.")
        print("Commands:")
        print("  !read <channel_id> [days=2] - Read messages from Discord channel")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                # Handle Discord read command
                if user_input.startswith("!read"):
                    parts = user_input.split()
                    channel_id = int(parts[1])
                    days = int(parts[2]) if len(parts) > 2 else 2
                    
                    messages = await self.read_channel(channel_id, days)
                    print(f"\nFound {len(messages)} messages")
                    response = await self.process_message(
                        f"Analyze these Discord messages and provide a summary: {messages}"
                    )
                else:
                    response = await self.process_message(user_input)
                    
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[Error] {str(e)}")
        
        await self.stop()
        print("\nChat session ended. Goodbye! ✨")

    async def stop(self) -> None:
        """Stop the runtime and cleanup Discord client."""
        if self.discord_client:
            await self.discord_client.close()