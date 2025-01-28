"""Example of using the local runtime system for cli based chatbots."""

import asyncio
import logging
from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.tools.image import ImageGenerationTool
from alchemist.ai.base.logging import configure_logging, LogLevel, LogComponent
from dotenv import load_dotenv

# Configure logging
# logger = logging.getLogger(__name__)

# Define a production-ready assistant persona
ASSISTANT = PersonaConfig(
    id="assistant-v1",
    name="Assistant",
    nickname="Assist",
    bio="I am a helpful assistant designed for production environments.",
    lore=[],
    style={
        "all": [
            "Clear and concise communication",
            "Professional tone",
            "Helpful and informative responses"
        ]
    },
    personality={}
)

async def run_with_runtime():
    """Example of using LocalRuntime for a more configured experience."""
    # Create runtime configuration
    config = RuntimeConfig(
        provider="openpipe",
        model="openpipe:ken0-llama31-8B-instruct",
        persona=ASSISTANT,
        tools=[ImageGenerationTool],
        platform_config={
            "prompt_prefix": "You: ",
            "response_prefix": "Assistant: "
        }
    )
    
    # Initialize and start local runtime
    runtime = LocalRuntime(config)
    print("\nChat using runtime (Ctrl+C to exit)")
    print("Try asking for calculations or image generation!")
    print("-----------------------------------")
    
    await runtime.start()

async def main():
    """Run both chat examples."""
    # # Set up logging
    # configure_logging(
    #     default_level=LogLevel.DEBUG,  # Set to DEBUG for maximum detail
    #     component_levels={
    #         LogComponent.RUNTIME: LogLevel.DEBUG,
    #         LogComponent.AGENT: LogLevel.DEBUG
    #     }
    # )
    
    load_dotenv()  # Load environment variables
    await run_with_runtime()  # Using LocalRuntime

if __name__ == "__main__":
    asyncio.run(main())