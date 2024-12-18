"""
Utility script to generate Discord bot invite URL with proper permissions.
"""

from discord import Permissions
import os
from dotenv import load_dotenv

def generate_invite_url():
    """
    Generates a Discord OAuth2 URL for bot invitation with required permissions.
    
    Returns:
        str: The complete OAuth2 URL for bot invitation
    """
    load_dotenv()
    
    CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
    
    if not CLIENT_ID:
        raise ValueError("DISCORD_CLIENT_ID not found in environment variables")
    
    # Define required permissions
    permissions = Permissions(
        send_messages=True,
        send_messages_in_threads=True,
        create_public_threads=True,
        manage_threads=True,
        send_tts_messages=True,
        embed_links=True,
        attach_files=True,
        read_message_history=True,
        add_reactions=True,
        mention_everyone=True
    )

    # Generate the URL
    base_url = "https://discord.com/api/oauth2/authorize"
    url = f"{base_url}?client_id={CLIENT_ID}&permissions={permissions.value}&scope=bot%20applications.commands"
    
    return url

if __name__ == "__main__":
    try:
        invite_url = generate_invite_url()
        print("\nUse this URL to invite the bot to your server:")
        print(invite_url)
    except Exception as e:
        print(f"Error generating invite URL: {e}") 