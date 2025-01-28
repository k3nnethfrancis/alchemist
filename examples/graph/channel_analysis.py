"""Simple Discord channel analysis workflow.

This example demonstrates a basic graph workflow that:
1. Collects messages from a Discord channel
2. Analyzes the content using LLM
3. Formats the results

The workflow is intentionally simplified to focus on:
- Proper state management
- Content formatting
- Prompt templates
- Error handling
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from alchemist.ai.graph.base import Graph, NodeContext, NodeState, Node
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.tools.discord_toolkit import DiscordTools
from alchemist.ai.prompts.base import prompt_template

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChannelCollectorNode(Node):
    """Collects messages from a Discord channel."""
    
    id: str = "collect"
    channel_name: str = Field(description="Name of channel to collect from")
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the collection of Discord messages.
        
        Args:
            state: Current node state
            
        Returns:
            ID of next node to execute
        """
        logger.info(f"🔍 Collecting messages from #{self.channel_name}")
        
        try:
            # Initialize Discord tools
            tools = DiscordTools(
                channels={"ai-news": "1326422578340036689"},  # AI News channel
                categories={}
            )
            
            # Get recent messages
            messages = await tools.read_channel(
                channel_name=self.channel_name,
                after=datetime.now() - timedelta(days=1),
                limit=10
            )
            
            logger.info(f"📥 Collected {len(messages)} messages")
            logger.debug(f"Message structure: {messages[0] if messages else 'No messages'}")
            
            # Update state with results
            state.results[self.id] = {
                "messages": messages,
                "count": len(messages)
            }
            
            return self.next_nodes.get("next")
            
        except Exception as e:
            logger.error(f"❌ Error collecting messages: {str(e)}", exc_info=True)
            state.errors[self.id] = str(e)
            return None

@prompt_template
def analysis_prompt(messages: str) -> str:
    """Template for analyzing Discord messages."""
    return """Analyze the following Discord messages and provide a summary:

Messages:
{messages}

Please provide:
1. Key topics discussed
2. Important links shared
3. Overall sentiment
4. Notable trends or patterns

Format your response in markdown."""

class ChannelAnalysisNode(LLMNode):
    """Analyzes collected Discord messages."""
    
    id: str = "analyze"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the analysis of collected messages.
        
        Args:
            state: Current node state with collected messages
            
        Returns:
            ID of next node to execute
        """
        logger.info("📊 Starting message analysis")
        
        try:
            # Get messages from previous state
            messages = state.results.get("collect", {}).get("messages", [])
            if not messages:
                raise ValueError("No messages found in state")
                
            logger.debug(f"Processing {len(messages)} messages")
            
            # Format messages for analysis
            formatted_messages = []
            for msg in messages:
                content = msg.get("content", "").strip()
                embeds = msg.get("embeds", [])
                
                if content:
                    formatted_messages.append(f"Content: {content}")
                
                for embed in embeds:
                    if title := embed.get("title"):
                        formatted_messages.append(f"Title: {title}")
                    if desc := embed.get("description"):
                        formatted_messages.append(f"Description: {desc}")
                    if url := embed.get("url"):
                        formatted_messages.append(f"URL: {url}")
            
            # Generate analysis prompt
            prompt = analysis_prompt(messages="\n".join(formatted_messages))
            logger.debug(f"Analysis prompt:\n{prompt}")
            
            # Get LLM response
            response = await self.llm.generate(prompt)
            logger.info("✅ Analysis complete")
            
            # Update state
            state.results[self.id] = {
                "analysis": response,
                "message_count": len(messages)
            }
            
            return self.next_nodes.get("next")
            
        except Exception as e:
            logger.error(f"❌ Error in analysis: {str(e)}", exc_info=True)
            state.errors[self.id] = str(e)
            return None

class ResultFormatterNode(Node):
    """Formats the analysis results."""
    
    id: str = "format"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Format the analysis results.
        
        Args:
            state: Current node state with analysis
            
        Returns:
            ID of next node to execute
        """
        logger.info("📝 Formatting results")
        
        try:
            # Get analysis from state
            analysis = state.results.get("analyze", {}).get("analysis")
            if not analysis:
                raise ValueError("No analysis found in state")
            
            # Format results
            formatted = f"""# Discord Channel Analysis

## Overview
- Channel: #ai-news
- Messages Analyzed: {state.results['analyze']['message_count']}
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis
{analysis}
"""
            logger.debug(f"Formatted output:\n{formatted}")
            
            # Update state
            state.results[self.id] = {
                "formatted": formatted
            }
            
            return self.next_nodes.get("next")
            
        except Exception as e:
            logger.error(f"❌ Error formatting results: {str(e)}", exc_info=True)
            state.errors[self.id] = str(e)
            return None

async def main():
    """Run the channel analysis workflow."""
    
    # Create nodes
    collector = ChannelCollectorNode(channel_name="ai-news")
    analyzer = ChannelAnalysisNode()
    formatter = ResultFormatterNode()
    
    # Create graph
    graph = Graph()
    
    # Add nodes and edges
    graph.add_node(collector)
    graph.add_node(analyzer)
    graph.add_node(formatter)
    
    graph.add_edge(collector.id, "next", analyzer.id)
    graph.add_edge(analyzer.id, "next", formatter.id)
    
    # Add entry point
    graph.add_entry_point("start", collector.id)
    
    # Create context and state
    context = NodeContext()
    state = NodeState(context=context)
    
    # Run graph
    logger.info("🚀 Starting channel analysis workflow")
    state = await graph.run("start", state)
    
    if state.errors:
        logger.error("❌ Workflow completed with errors")
        for node_id, error in state.errors.items():
            logger.error(f"Node {node_id}: {error}")
    else:
        logger.info("✅ Workflow completed successfully")
        print("\nAnalysis Results:")
        print(state.results["format"]["formatted"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 