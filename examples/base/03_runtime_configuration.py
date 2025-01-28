"""Example of using the runtime system for production deployment.

This example demonstrates using RuntimeConfig for production environments:
1. Advanced provider and model configuration
2. Session management and tracking
3. Platform-specific optimizations
4. Production-ready logging and monitoring

Best for:
- Production deployments
- Multi-model applications
- Session tracking
- Platform integration
"""

import asyncio
from alchemist.ai.base.runtime import RuntimeConfig, LocalRuntime
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.tools.calculator import CalculatorTool
from alchemist.ai.base.logging import configure_logging, LogLevel, LogComponent

# Define a production-ready assistant persona
PRODUCTION_ASSISTANT = {
    "id": "prod-assistant-v1",
    "name": "ProdGPT",
    "nickname": "Prod",
    "bio": """I am a production-grade assistant optimized for reliable performance.
I demonstrate runtime configuration and session management capabilities.""",
    "personality": {
        "traits": {
            "neuroticism": 0.1,      # Highly stable
            "extraversion": 0.5,      # Balanced interaction
            "openness": 0.7,         # Adaptable but consistent
            "agreeableness": 0.8,    # Professional and helpful
            "conscientiousness": 1.0  # Maximum reliability
        },
        "stats": {
            "intelligence": 0.9,      # High capability
            "wisdom": 0.9,           # Excellent judgment
            "charisma": 0.7,         # Professional communication
            "authenticity": 1.0,     # Complete transparency
            "adaptability": 0.8,     # Flexible to needs
            "reliability": 1.0       # Maximum reliability
        }
    },
    "lore": [
        "Designed for production use",
        "Optimized for reliability",
        "Handles high-load scenarios",
        "Maintains consistent performance",
        "Provides detailed session tracking"
    ],
    "style": {
        "all": [
            "Uses consistent formatting",
            "Provides detailed responses",
            "Maintains professional tone",
            "Includes session context",
            "Reports performance metrics"
        ],
        "chat": [
            "Acknowledges session state",
            "Uses structured responses",
            "Provides progress updates",
            "Handles errors gracefully",
            "Maintains context awareness"
        ]
    }
}

async def main():
    """Run a production-configured session."""
    # Configure comprehensive logging
    configure_logging(
        default_level=LogLevel.INFO,
        component_levels={
            LogComponent.AGENT: LogLevel.DEBUG,
            LogComponent.RUNTIME: LogLevel.DEBUG,
            LogComponent.TOOLS: LogLevel.INFO
        }
    )
    
    # Create production runtime configuration
    config = RuntimeConfig(
        # Provider configuration
        provider="openai",
        model="claude-3-5-sonnet-20240620",
        
        # Core configuration
        persona=PRODUCTION_ASSISTANT,
        tools=[CalculatorTool],
        
        # Platform-specific settings
        platform_config={
            # Model parameters
            "temperature": 0.5,        # Balanced creativity/consistency
            "max_tokens": 1000,        # Reasonable response length
            "top_p": 0.9,             # High-quality sampling
            
            # Runtime settings
            "timeout": 30,            # 30-second timeout
            "retry_attempts": 3,      # Retry failed calls
            "cache_responses": True,   # Enable response caching
            
            # Session configuration
            "track_metrics": True,     # Enable performance tracking
            "log_level": "DEBUG"       # Detailed logging
        }
    )
    
    # Initialize runtime with production config
    runtime = LocalRuntime(config)
    
    print("\nInitialized Production Runtime")
    print("Features enabled:")
    print("- Session tracking")
    print("- Performance monitoring")
    print("- Response caching")
    print("- Error recovery")
    print("\nTry asking:")
    print("- What capabilities do you have?")
    print("- Perform some calculations")
    print("- Tell me about the current session")
    print("\nType 'exit' or 'quit' to end")
    print("-" * 50)
    
    # Start runtime with full features
    await runtime.start()
    
    while True:
        query = input("\n(Production): ")
        if query.lower() in ["exit", "quit"]:
            await runtime.stop()
            break
            
        # Process with full runtime features
        result = await runtime.process_message(query)
        print(f"\n(Assistant): {result}")
    
    print("\nSession ended - Runtime stopped")

if __name__ == "__main__":
    asyncio.run(main()) 