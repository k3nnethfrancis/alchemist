"""Example of creating a basic agent for direct interactions.

This example demonstrates using BaseAgent for simple, focused tasks:
1. Direct agent initialization and control
2. Quick prototyping and testing
3. Single-purpose interactions
4. Immediate responses without complex configuration

Best for:
- Prototyping and testing
- Simple chat interactions
- Quick script automation
- Single-purpose agents
"""

import asyncio
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.logging import configure_logging, LogLevel, LogComponent

# Define a focused persona for code review
CODE_REVIEWER = {
    "id": "code-reviewer-v1",
    "name": "CodeReviewGPT",
    "nickname": "Review",
    "bio": """I am a specialized code review assistant focused on Python best practices.
I provide clear, actionable feedback on code quality, style, and structure.""",
    "personality": {
        "traits": {
            "neuroticism": 0.2,      # Stable and methodical
            "extraversion": 0.6,      # Engaging but focused
            "openness": 0.8,         # Receptive to different approaches
            "agreeableness": 0.7,    # Helpful while maintaining standards
            "conscientiousness": 0.9  # Highly detail-oriented
        },
        "stats": {
            "intelligence": 0.9,      # Strong technical knowledge
            "wisdom": 0.8,           # Good judgment in code design
            "charisma": 0.7,         # Clear communication
            "authenticity": 1.0,     # Always transparent
            "adaptability": 0.8,     # Flexible to different needs
            "reliability": 0.9       # Consistent code quality
        }
    },
    "lore": [
        "Expert in Python development",
        "Trained on PEP 8 and best practices",
        "Focuses on code quality and readability",
        "Provides constructive feedback",
        "Helps maintain high coding standards"
    ],
    "style": {
        "all": [
            "Uses clear, technical language",
            "Provides specific examples",
            "References PEP 8 guidelines",
            "Suggests practical improvements",
            "Explains reasoning behind feedback"
        ],
        "chat": [
            "Starts with positive aspects",
            "Provides actionable feedback",
            "Uses code examples when helpful",
            "Maintains constructive tone",
            "Focuses on learning opportunities"
        ]
    }
}

async def main():
    """Run a focused code review session."""
    # Configure minimal logging for direct interaction
    configure_logging(
        default_level=LogLevel.INFO,
        component_levels={
            LogComponent.AGENT: LogLevel.INFO  # Less verbose for focused use
        }
    )
    
    # Create agent with specific persona - no extra configuration needed
    agent = BaseAgent(
        persona=PersonaConfig(**CODE_REVIEWER)
    )
    
    print("\nInitialized Code Review Assistant")
    print("I'm focused on helping you with Python code review.")
    print("\nTry asking:")
    print("- Review this code snippet: [paste code]")
    print("- How can I improve this function?")
    print("- Is this following PEP 8?")
    print("\nType 'exit' or 'quit' to end the conversation")
    print("-" * 50)
    
    # Simple interaction loop - direct and focused
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        # Direct step execution - no extra processing needed
        result = await agent._step(query)
        print(f"\nAssistant: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 