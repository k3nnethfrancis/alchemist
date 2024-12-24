"""# AI Examples

This directory contains example applications demonstrating different ways to use the AI framework.

## Directory Structure

- `chat/`: Examples of local chat applications
  - `local_chat.py`: Simple command-line chat interface
  
- `discord/`: Discord bot examples
  - `chat_bot.py`: Basic Discord chat bot using the chat framework
  - `eliza_bot.py`: Advanced Discord bot using the Eliza framework for proactive engagement

## Running Examples

### Local Chat
```bash
python -m ai.examples.chat.local_chat
```

### Discord Bots
First set up your Discord bot token in `.env`:
```
DISCORD_TOKEN=your_token_here
```

Then run either:
```bash
# Basic chat bot
python -m ai.examples.discord.chat_bot

# Advanced Eliza bot
python -m ai.examples.discord.eliza_bot
```
""" 