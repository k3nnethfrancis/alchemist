"""Discord chatbot example.

This example demonstrates a simple Discord bot that:
1. Uses a custom personality (Ken-E)
2. Responds to messages in a configured channel
3. Requires run_bot.py to be running first

Usage:
    1. Start the Discord service: python -m examples.discord.run_bot
    2. Then run this chatbot: python -m examples.discord.chatbot
"""

import os
import asyncio
from dotenv import load_dotenv
import discord

from alchemist.ai.prompts.persona import KEN_E
from alchemist.extensions.discord.runtime import DiscordRuntimeConfig, DiscordChatRuntime
from alchemist.ai.base.runtime import RuntimeConfig

async def handle_message(message: discord.Message, runtime: DiscordChatRuntime):
    """Handle incoming Discord messages.
    
    Args:
        message: The Discord message to process
        runtime: The Discord runtime instance
    """
    # Only process messages in configured channels
    if str(message.channel.id) not in runtime.config.channel_ids:
        return
        
    # Only process messages that mention the bot
    if not message.mentions or runtime.client.user not in message.mentions:
        return
        
    # Remove the bot mention from the message
    content = message.content.replace(f'<@{runtime.client.user.id}>', '').strip()
    
    try:
        # Process the message using the runtime's process_message method
        response = await runtime.process_message(content)
        
        # Send the response back to Discord
        await message.channel.send(response)
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        await message.channel.send("Sorry, I encountered an error processing your message.")

async def main():
    """Run the Discord chatbot."""
    # Load environment variables
    load_dotenv()
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("Error: DISCORD_BOT_TOKEN not set in .env file")
        return

    # Configure Discord runtime with both Discord and runtime configs
    discord_config = DiscordRuntimeConfig(
        bot_token=token,
        channel_ids=["1318659602115592204"],  # agent-sandbox channel
        runtime_config=RuntimeConfig(
            provider="openpipe",
            model="openpipe:ken0-llama31-8B-instruct",
            persona=KEN_E,
        )
    )
    
    # Create and start Discord runtime
    runtime = DiscordChatRuntime(config=discord_config)
    
    # Add message handler
    runtime.add_message_handler(lambda msg: handle_message(msg, runtime))
    
    # Start the runtime
    await runtime.start()

    print("\nDiscord chatbot running with Ken-0 model!")
    print("Press Ctrl+C to exit")

    try:
        await asyncio.Future()  # run forever
    except KeyboardInterrupt:
        await runtime.stop()
        print("\nChatbot stopped")

if __name__ == "__main__":
    asyncio.run(main()) 