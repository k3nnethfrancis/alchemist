"""OpenPipe client configuration."""

from openpipe import OpenAI
from typing import Optional
from dotenv import load_dotenv; load_dotenv()

def get_openpipe_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Creates an OpenPipe client instance."""
    return OpenAI(
        api_key=api_key,
        openpipe={
            "api_key": api_key,
            "base_url": base_url or "https://api.openpipe.ai/api/v1"
        }
    ) 