# eliza_agent/providers/providers.py

import asyncio

class MessageHistoryProvider:
    def __init__(self):
        # Initialize message history storage
        self.channel_histories = {}

    async def get_recent_messages(self, channel_id, limit=50):
        # Fetch recent messages for the channel
        # For now, return dummy data
        messages = self.channel_histories.get(channel_id, [])
        # Format messages as needed for the template
        formatted_messages = '\n'.join(messages[-limit:])
        return formatted_messages

    def add_message(self, channel_id, message):
        # Add a new message to the history
        if channel_id not in self.channel_histories:
            self.channel_histories[channel_id] = []
        self.channel_histories[channel_id].append(message)

class AgentKnowledgeProvider:
    def __init__(self, agent_profile):
        self.agent_profile = agent_profile

    def get_bio(self):
        return self.agent_profile.get('bio', '')

    def get_lore(self):
        return self.agent_profile.get('lore', [])