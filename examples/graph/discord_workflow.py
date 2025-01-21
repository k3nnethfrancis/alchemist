"""Example of a proactive Discord bot using graph nodes."""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, Field

# Add parent directory to path
file = os.path.abspath(__file__)
parent = os.path.dirname(os.path.dirname(os.path.dirname(file)))
sys.path.insert(0, parent)

from alchemist.ai.graph.base import Graph, NodeState, NodeContext
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.graph.nodes.decisions import BinaryDecisionNode
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.persona import AUG_E
from alchemist.core.extensions.discord.runtime import DiscordRuntimeConfig, DiscordRuntime

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "discord_workflow.log"),
        logging.StreamHandler()
    ]
)

# Set logging levels for noisy modules
logging.getLogger("discord").setLevel(logging.INFO)
logging.getLogger("websockets").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Structured Output Models
class DiscordMessage(BaseModel):
    """Model for a Discord message."""
    content: str
    clean_content: str
    timestamp: float
    author: Dict[str, Any]
    channel: Dict[str, Any]
    urls: List[str] = Field(default_factory=list)
    embeds: List[Dict[str, Any]] = Field(default_factory=list)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    edited_timestamp: Optional[float] = None
    reference: Optional[Dict[str, Any]] = None

class EngagementTrigger(BaseModel):
    """Model for an engagement trigger."""
    type: str = Field(..., description="Type of trigger (greeting, community_call, topic, conversation)")
    content: str = Field(..., description="The content that triggered engagement")
    priority: int = Field(..., description="Priority level (1-5, 1 being highest)")
    must_engage: bool = Field(..., description="Whether this trigger requires mandatory engagement")

class ConversationState(BaseModel):
    """Model for conversation state analysis."""
    activity_level: str = Field(..., description="Activity level (active, stale, recent)")
    last_message_time: datetime
    participant_count: int
    recent_topics: List[str] = Field(default_factory=list)

class AnalysisResult(BaseModel):
    """Model for message analysis results."""
    triggers: List[EngagementTrigger] = Field(default_factory=list)
    conversation_state: ConversationState
    recommended_action: str = Field(..., description="MUST_ENGAGE, SHOULD_ENGAGE, MAY_ENGAGE, or NO_ENGAGEMENT")
    messages: List[DiscordMessage]

class MessageAnalysisNode(LLMNode):
    """Node that analyzes recent messages for engagement opportunities."""
    
    async def process(self, state: NodeState) -> str:
        """Analyze recent messages and identify engagement opportunities."""
        try:
            # Get recent messages from context
            messages = state.context.metadata.get("recent_messages", [])
            if not messages:
                logger.info("[Analysis] No messages found to analyze")
                return "skip"
                
            # Convert raw messages to DiscordMessage models
            discord_messages = []
            for msg in messages:
                if msg["author"].get("bot", False):
                    logger.debug(f"[Analysis] Skipping bot message from {msg['author']['name']}")
                    continue
                discord_messages.append(DiscordMessage(**msg))
            
            if not discord_messages:
                logger.info("[Analysis] No human messages found to analyze")
                return "skip"
                
            logger.info(f"[Analysis] Analyzing {len(discord_messages)} messages")
            
            # Format messages for LLM analysis
            formatted_messages = []
            for msg in discord_messages:
                formatted_messages.append(
                    f"[{datetime.fromtimestamp(msg.timestamp).strftime('%H:%M:%S')}] "
                    f"{msg.author['name']}: {msg.content}"
                )
            
            # Analyze messages
            prompt = f"""Analyze these recent messages and identify engagement opportunities:

{chr(10).join(formatted_messages)}

Consider these engagement triggers:
1. Direct greetings or questions (e.g. "hello", "anyone there", "yo") - MUST engage
2. Community calls (e.g. "agents unite", "calling all AI friends") - MUST engage
3. Conversation starters or topics I could contribute to
4. Signs of active conversation that I could naturally join

Format your response as a JSON object with this EXACT structure:
{{
    "triggers": [
        {{
            "type": "greeting|community_call|topic|conversation",
            "content": "exact content that triggered",
            "priority": 1-5 (1 highest),
            "must_engage": true|false
        }}
    ],
    "conversation_state": {{
        "activity_level": "active|stale|recent",
        "last_message_time": "ISO datetime",
        "participant_count": number,
        "recent_topics": ["topic1", "topic2"]
    }},
    "recommended_action": "MUST_ENGAGE|SHOULD_ENGAGE|MAY_ENGAGE|NO_ENGAGEMENT"
}}"""

            response = await self.agent._step(prompt)
            logger.info(f"[Analysis] Analysis result:\n{response}")
            
            # Parse response into structured output
            analysis = AnalysisResult.model_validate_json(response)
            
            # Store analysis in results
            state.results[self.id] = {
                "analysis": analysis.model_dump(),
                "messages": [msg.model_dump() for msg in discord_messages]
            }
            
            return "should_engage"
            
        except Exception as e:
            logger.error(f"[Analysis] Error in message analysis: {str(e)}")
            state.results["error"] = str(e)
            return "skip"

class EngagementResponse(BaseModel):
    """Model for engagement response."""
    message: str = Field(..., description="The message to send")
    tone: str = Field(..., description="The tone of the message (casual, friendly, professional)")
    referenced_content: Optional[str] = Field(None, description="Content being referenced/responded to")
    engagement_type: str = Field(..., description="Type of engagement (greeting, community, topical, conversational)")

class EngagementNode(LLMNode):
    """Node that generates and sends engagement responses."""
    
    async def process(self, state: NodeState) -> None:
        """Generate and send an engagement response."""
        try:
            # Get analysis from previous node
            analysis_node_id = [n for n in state.context.graph.nodes if isinstance(n, MessageAnalysisNode)][0].id
            analysis_results = state.results[analysis_node_id]
            
            if not analysis_results:
                logger.warning("[Engage] No analysis results found")
                return None
                
            analysis = AnalysisResult(**analysis_results["analysis"])
            logger.info(f"[Engage] Generating response based on analysis: {analysis.model_dump_json(indent=2)}")
            
            # Format messages for context
            messages = [DiscordMessage(**msg) for msg in analysis_results["messages"]]
            formatted_messages = []
            for msg in messages:
                formatted_messages.append(
                    f"[{datetime.fromtimestamp(msg.timestamp).strftime('%H:%M:%S')}] "
                    f"{msg.author['name']}: {msg.content}"
                )
            
            # Generate response
            prompt = f"""Based on these recent messages and analysis, generate an engaging response:

Recent Messages:
{chr(10).join(formatted_messages)}

Analysis:
- Triggers: {', '.join(f"{t.type} ({t.content})" for t in analysis.triggers)}
- Activity Level: {analysis.conversation_state.activity_level}
- Topics: {', '.join(analysis.conversation_state.recent_topics)}

Format your response as a JSON object with this EXACT structure:
{{
    "message": "your response message",
    "tone": "casual|friendly|professional",
    "referenced_content": "content being referenced (optional)",
    "engagement_type": "greeting|community|topical|conversational"
}}

Guidelines:
1. Keep responses casual and friendly
2. Match the energy level of the conversation
3. Reference specific content when appropriate
4. Use emojis sparingly but effectively"""

            response = await self.agent._step(prompt)
            logger.info(f"[Engage] Generated response:\n{response}")
            
            # Parse and validate response
            engagement = EngagementResponse.model_validate_json(response)
            
            # Send message using runtime
            runtime = state.context.metadata.get("runtime")
            if runtime and hasattr(runtime, "send_message"):
                channel_id = messages[-1].channel["id"]
                await runtime.send_message(channel_id, engagement.message)
                logger.info(f"[Engage] Sent message to channel {channel_id}")
            else:
                logger.error("[Engage] No runtime available to send message")
            
            # Store engagement in results
            state.results[self.id] = engagement.model_dump()
            
            return None  # End workflow
            
        except Exception as e:
            logger.error(f"[Engage] Error in engagement: {str(e)}")
            return None  # End workflow on error

class EnhancedBinaryDecisionNode(LLMNode):
    """Node that makes a binary decision about whether to engage based on message analysis."""
    
    async def process(self, state: NodeState) -> str:
        """Process the analysis and decide whether to engage."""
        try:
            # Get analysis from previous node
            analysis_node_id = [n for n in state.context.graph.nodes if isinstance(n, MessageAnalysisNode)][0].id
            analysis_results = state.results[analysis_node_id]
            
            if not analysis_results:
                logger.warning("[Decision] No analysis results found")
                return "no"
                
            analysis = AnalysisResult(**analysis_results["analysis"])
            logger.info(f"[Decision] Analyzing results: {analysis.model_dump_json(indent=2)}")
            
            # Check for MUST_ENGAGE triggers
            if analysis.recommended_action == "MUST_ENGAGE":
                logger.info("[Decision] Found MUST_ENGAGE trigger - proceeding to engage")
                return "yes"
                
            # Check for SHOULD_ENGAGE with high priority triggers
            if analysis.recommended_action == "SHOULD_ENGAGE" and any(t.priority <= 2 for t in analysis.triggers):
                logger.info("[Decision] Found high priority SHOULD_ENGAGE trigger - proceeding to engage")
                return "yes"
                
            # For MAY_ENGAGE, use LLM to make final decision
            if analysis.recommended_action == "MAY_ENGAGE":
                prompt = f"""Based on this conversation analysis, should I engage? Consider:
1. Triggers found: {', '.join(f"{t.type} ({t.content})" for t in analysis.triggers)}
2. Activity level: {analysis.conversation_state.activity_level}
3. Participant count: {analysis.conversation_state.participant_count}
4. Recent topics: {', '.join(analysis.conversation_state.recent_topics)}

Respond with ONLY 'yes' or 'no'."""

                response = await self.agent._step(prompt)
                response = response.strip().lower()
                
                if response not in ["yes", "no"]:
                    logger.warning(f"[Decision] Invalid response '{response}' - defaulting to yes")
                    return "yes"
                    
                logger.info(f"[Decision] LLM decided: {response}")
                return response
                
            # Default to no engagement
            logger.info("[Decision] No compelling reason to engage")
            return "no"
            
        except Exception as e:
            logger.error(f"[Decision] Error in decision making: {str(e)}")
            return "no"

async def create_discord_workflow() -> Graph:
    """Create a workflow for Discord engagement."""
    logger.info("Creating Discord engagement workflow")
    
    # Create agent
    agent = BaseAgent(
        provider="openpipe",
        model="gpt-4o-mini",
        persona=AUG_E,
        tools=[]
    )
    
    # Create nodes
    analyze = MessageAnalysisNode(
        id="analyze",
        name="analyze",
        agent=agent,
        next_nodes={
            "should_engage": "should_engage",
            "skip": None  # End workflow
        }
    )
    
    should_engage = EnhancedBinaryDecisionNode(
        id="should_engage",
        name="should_engage",
        agent=agent,
        next_nodes={
            "yes": "engage",
            "no": None  # End workflow
        }
    )
    
    engage = EngagementNode(
        id="engage",
        name="engage",
        agent=agent,
        next_nodes={}  # All paths end workflow
    )
    
    # Create graph
    graph = Graph()
    graph.add_node(analyze)
    graph.add_node(should_engage)
    graph.add_node(engage)
    
    logger.info("Discord workflow created successfully")
    return graph

async def main():
    """Initialize and run the proactive Discord bot."""
    try:
        # Load environment variables
        load_dotenv()
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            logger.error("DISCORD_BOT_TOKEN not set in .env file")
            sys.exit(1)
        
        # Create workflow
        workflow = await create_discord_workflow()
        
        # Configure runtime
        config = DiscordRuntimeConfig(
            provider="openpipe",
            model="gpt-4o-mini",
            persona=AUG_E,
            platform_config={
                "workflow": workflow,  # Add our workflow
                "check_interval": 60,  # Check for engagement opportunities every 60 seconds
            },
            bot_token=token,
            channel_ids=["1318659602115592204"]  # agent-sandbox channel
        )
        
        # Create and start runtime
        runtime = DiscordRuntime(config=config)
        
        logger.info("Starting Discord workflow bot...")
        await runtime.start()
        
        # Keep the bot running
        try:
            await asyncio.Future()  # run forever
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await runtime.stop()
            
    except Exception as e:
        logger.error(f"Error in Discord workflow: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())