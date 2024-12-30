"""Example of a proactive Discord bot using graph nodes."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import asyncio
import discord

from alchemist.ai.graph.base import Graph, NodeState, NodeContext
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.graph.nodes.decisions import BinaryDecisionNode
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.persona import AUG_E
from alchemist.core.extensions.discord.runtime import DiscordRuntime
from alchemist.ai.base.runtime import RuntimeConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageAnalysisNode(LLMNode):
    """Node that analyzes recent messages for engagement opportunities."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Analyze messages and identify engagement opportunities."""
        try:
            messages = state.context.metadata.get("recent_messages", [])
            if not messages:
                logger.debug("No messages to analyze")
                return "skip"
                
            # Format messages for analysis
            formatted_messages = "\n".join([
                f"[{msg['author']['name']}]: {msg['content']}"
                for msg in messages
            ])
            
            # Analyze conversation
            prompt = f"""Review these recent messages and identify if there's an opportunity for meaningful engagement:

            Recent Messages:
            {formatted_messages}

            Consider:
            1. Are there questions that haven't been answered?
            2. Are there topics where you could add value?
            3. Is there a natural opening for conversation?
            4. Would your intervention be helpful or disruptive?

            Provide your analysis and reasoning."""
            
            response = await self.agent.get_response(prompt)
            
            # Store analysis
            state.results[self.id] = {
                "analysis": response,
                "messages": messages
            }
            
            return "should_engage"  # Changed from "default" to explicit next node
            
        except Exception as e:
            logger.error(f"Error in message analysis: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            return "error"

class EngagementNode(LLMNode):
    """Node that generates contextual responses for proactive engagement."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Generate appropriate response for the conversation."""
        try:
            analysis = state.results["analyze"]["analysis"]
            messages = state.results["analyze"]["messages"]
            
            prompt = f"""Based on the conversation analysis, craft an engaging response:

            Analysis: {analysis}
            
            Guidelines:
            1. Be natural and conversational
            2. Add value to the discussion
            3. Maintain AUG_E's personality
            4. Be mindful of conversation flow
            
            Craft your response:"""
            
            response = await self.agent.get_response(prompt)
            
            # Store response
            state.results[self.id] = {
                "response": response,
                "channel_id": messages[0]["channel"]["id"]  # Save channel for sending
            }
            
            return "default"
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            return "error"

async def create_discord_workflow() -> Graph:
    """Create Discord engagement workflow."""
    logger.info("Creating Discord engagement workflow...")
    
    # Create agent with AUG_E persona
    agent = BaseAgent(provider="openpipe", persona=AUG_E)
    
    # Create nodes
    analyze = MessageAnalysisNode(
        id="analyze",
        agent=agent,
        next_nodes={
            "should_engage": "should_engage",
            "skip": None,
            "error": None
        }
    )
    
    should_engage = BinaryDecisionNode(
        id="should_engage",
        agent=agent,
        prompt="""Based on this analysis, should I engage with the conversation?
        
        Analysis: {analyze[analysis]}
        
        Consider:
        1. Is there a clear opportunity to add value?
        2. Would my intervention be natural and helpful?
        3. Has enough time passed since my last message?
        
        Respond with only 'yes' or 'no'.""",
        next_nodes={
            "yes": "engage",
            "no": None,
            "error": None
        }
    )
    
    engage = EngagementNode(
        id="engage",
        agent=agent,
        next_nodes={
            "default": None,
            "error": None
        }
    )
    
    # Create and validate graph
    graph = Graph()
    graph.add_node(analyze)
    graph.add_node(should_engage)
    graph.add_node(engage)
    
    # Add entry point
    graph.add_entry_point("main", "analyze")
    
    return graph

async def main():
    """Initialize and run the proactive Discord bot."""
    try:
        # Load token from environment
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN not set")
            
        # Configure runtime with required intents
        intents = discord.Intents.default()
        intents.message_content = True  # For message content
        intents.guild_messages = True   # For guild messages
        intents.guilds = True          # For guild access
        
        config = RuntimeConfig(
            provider="openpipe",
            persona=AUG_E,
            platform_config={
                "intents": intents,
                "activity_type": "watching",
                "activity_name": "the conversation"
            }
        )
        
        # Create workflow
        workflow = await create_discord_workflow()
        
        # Initialize and start runtime with workflow
        runtime = DiscordRuntime(
            token=token, 
            config=config,
            workflow=workflow
        )
        await runtime.start()
        
    except Exception as e:
        logger.error(f"Error running Discord bot: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())