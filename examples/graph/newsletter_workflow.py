"""Newsletter Automata: A graph-based workflow for generating AI newsletters from Discord content."""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import asyncio
import discord
from pydantic import Field

from alchemist.ai.graph.base import Graph, NodeState, NodeContext
from alchemist.ai.graph.nodes.base import LLMNode
from alchemist.ai.graph.nodes.decisions import BinaryDecisionNode
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.persona import AUG_E, KEN_E
from alchemist.core.extensions.discord.runtime import DiscordRuntime
from alchemist.ai.base.runtime import RuntimeConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONSTANTS
PERSONA = KEN_E
MESSAGES_TO_FETCH = 42
TARGET_AUDIENCE =   """
                    Technologists. Entrepreneurs. Developers. Creators. Founders. Business leaders. 
                    All interested in going deep on AI to enhance their work and business.
                    """       

# Newsletter Prompt Templates
NEWSLETTER_IDENTITY = f"""
Newsletter Identity:
- Title: "Improbable Automata: Field notes from the AI frontier"
- Subtitle: "(hallucinations may vary)"
- Target Audience: {TARGET_AUDIENCE}
"""

NEWSLETTER_STRUCTURE = """
Structure:
1. Synopsis
    - 1-3 passages summarizing the most interesting and relevant content from the week
    - Written in plain but engaging language
    - Draw excitement with compelling hooks
    - Free-flowing, natural style, keeping it interesting for readers
    - Embed links to best content

2. <custom title>
    - The first section after field notes is a curated list of the best content from the week, title is custom to the vibe of the week
    - We care about business relevance, technical insights, and practical applications
    - Flow is more structured for this section. use a mix of bullet points and short paragraphs to keep it interesting
    - We want to include relevant links in each section while maintaining flow

3. Closing Thoughts
   - Brief, impactful final words
   - Connect to broader themes
"""

WRITING_STYLE = """
Writing Style:
- Informative yet playful
- Technical concepts through clear metaphors
- Assume intelligence, explain complexity
- Balance wonder with practicality
- Use humor to illuminate, not distract

Voice:
- Knowledgeable but approachable
- Excited about possibilities
- Grounded in practical reality
- Speaking peer-to-peer with tech leaders
- Focus on business value and implementation
"""

ANALYSIS_PROMPT = f"""Analyze these Discord messages for Improbable Automata newsletter:
Messages: {{collect[content]}}
Date Range: {{collect[date_range]}}

Target Audience: {TARGET_AUDIENCE}

Consider:
1. Who does this content help?
2. Is this content highly technical or academic?
3. Does this content need practical explanation?
4. Are there ways we can think of making use of this technology or new development to improve or work or products?
5. Does this content point to any trends or interesting ideas?

For each shared content piece, consider the above and note your thoughts. 
Keep in mind the desire to make this content useful and actionable for the target audience.
Keep it technically interesting, but practical and insightful.
"""

DRAFT_PROMPT = f"""Generate an engaging field notes entry for Improbable Automata:

Analysis: {{analyze[response]}}

{NEWSLETTER_IDENTITY}
{NEWSLETTER_STRUCTURE}"""

def create_newsletter_prompt(additional_context: str = "") -> str:
    """
    Create a complete newsletter prompt with optional additional context.
    
    Args:
        additional_context (str): Additional instructions or context to include
    
    Returns:
        str: Complete formatted prompt
    """
    return f"""{NEWSLETTER_IDENTITY}
{NEWSLETTER_STRUCTURE}
{WRITING_STYLE}
{additional_context if additional_context else ""}"""

FORMAT_PROMPT = """Format this content into an engaging Improbable Automata newsletter:

Content: {draft}

{newsletter_template}

Important:
- Ensure ALL sections maintain relevant links
- Weekly Signal should reference specific examples from Field Notes
- Keep links natural within the flow of text"""

class ContentCollectorNode(LLMNode):
    """Node that collects and processes Discord messages."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process Discord messages into structured content."""
        try:
            # Get messages from runtime context
            messages = state.context.metadata.get("recent_messages", [])
            if not messages:
                logger.debug("No messages to analyze")
                return None
            
            # Process messages into structured format with enhanced link tracking
            processed_content = []
            link_collection = []  # Store all unique links with context
            
            for msg in messages:
                # Extract links from content and embeds
                content_links = []
                
                # Direct URLs in content
                if msg["urls"]:
                    for url in msg["urls"]:
                        content_links.append({
                            "url": url,
                            "context": msg["clean_content"],
                            "timestamp": msg["timestamp"],
                            "source": "direct_share"
                        })
                
                # Links from embeds
                if msg["embeds"]:
                    for embed in msg["embeds"]:
                        if embed["url"]:
                            content_links.append({
                                "url": embed["url"],
                                "title": embed["title"],
                                "description": embed["description"],
                                "timestamp": msg["timestamp"],
                                "source": "embed",
                                "image": embed["image"] if embed.get("image") else None
                            })
                
                # Add to global link collection
                link_collection.extend(content_links)
                
                # Add links to processed content
                content = {
                    "raw_content": msg["content"],
                    "clean_content": msg["clean_content"],
                    "timestamp": msg["timestamp"],
                    "author": msg["author"]["name"],
                    "urls": msg["urls"],
                    "embeds": [
                        {
                            "title": embed["title"],
                            "description": embed["description"],
                            "url": embed["url"],
                            "image": embed["image"],
                            "fields": embed["fields"]
                        }
                        for embed in msg["embeds"]
                    ],
                    "attachments": [
                        {
                            "filename": att["filename"],
                            "url": att["url"],
                            "content_type": att["content_type"]
                        }
                        for att in msg["attachments"]
                    ],
                    "edited": msg["edited_timestamp"] is not None,
                    "context": self._extract_context(msg),
                    "links": content_links
                }
                processed_content.append(content)
            
            # Store processed content with enhanced link data
            state.results[self.id] = {
                "content": processed_content,
                "total_messages": len(messages),
                "date_range": f"{datetime.fromtimestamp(min(m['timestamp'] for m in messages))} to {datetime.fromtimestamp(max(m['timestamp'] for m in messages))}",
                "link_collection": link_collection
            }
            
            return "analyze"
            
        except Exception as e:
            logger.error(f"Error collecting content: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            return None

    def _extract_context(self, msg: Dict[str, Any]) -> str:
        """Extract contextual information from message."""
        context_parts = []
        
        # Add embed information
        if msg["embeds"]:
            for embed in msg["embeds"]:
                if embed["title"]:
                    context_parts.append(f"Title: {embed['title']}")
                if embed["description"]:
                    context_parts.append(f"Description: {embed['description']}")
                if embed["fields"]:
                    for field in embed["fields"]:
                        context_parts.append(f"{field['name']}: {field['value']}")
        
        # Add attachment information
        if msg["attachments"]:
            for att in msg["attachments"]:
                context_parts.append(f"Attached {att['content_type']}: {att['filename']}")
        
        # Add reference information if it's a reply
        if msg.get("reference"):
            context_parts.append("In reply to previous message")
        
        return "\n".join(context_parts) if context_parts else ""

class NewsletterFormatterNode(LLMNode):
    """Node that formats content according to Improbable Automata style guide."""
    
    # Declare additional_context as a proper field
    additional_context: str = Field(default="")
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Format the newsletter draft according to style guidelines."""
        try:
            draft = state.results.get("enrich", {}).get("response")
            if not draft:
                logger.warning("No enriched content to format")
                return None
            
            # Create complete newsletter template with context
            newsletter_template = create_newsletter_prompt(self.additional_context)
            
            # Format according to style guide
            response = await self.agent.get_response(
                FORMAT_PROMPT.format(
                    draft=draft,
                    newsletter_template=newsletter_template
                )
            )
            
            state.results[self.id] = {"response": response}
            return None
            
        except Exception as e:
            logger.error(f"Error formatting newsletter: {str(e)}")
            return None

class LinkEnricherNode(LLMNode):
    """Node that enriches newsletter content with properly formatted links."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Enrich newsletter content with collected links."""
        try:
            draft = state.results.get("draft", {}).get("response")
            collected = state.results.get("collect", {})
            
            if not draft or not collected:
                logger.warning("Missing draft or collected content for enrichment")
                return None
            
            # Extract links from collected content
            links = []
            for message in collected.get("content", []):
                # Get links from message
                if message.get("urls"):
                    for url in message["urls"]:
                        link_info = {
                            "url": url,
                            "title": None,
                            "description": None
                        }
                        
                        # Try to get title/description from embeds
                        for embed in message.get("embeds", []):
                            if embed.get("url") == url:
                                link_info["title"] = embed.get("title")
                                link_info["description"] = embed.get("description")
                                break
                        
                        links.append(link_info)
            
            # Format links for prompt
            formatted_links = "\n".join([
                f"- {link['title'] or 'Shared Content'}: {link['url']}\n  "
                f"Context: {link['description'] or 'No description available'}"
                for link in links
            ])
            
            prompt = """Enrich this newsletter with relevant links from our collection:

Draft Content:
{draft}

Available Links:
{links}

Instructions:
1. Insert markdown-style links [text](url) naturally within the content
2. Ensure each major point or reference has its supporting link
3. Maintain the natural flow and readability
4. Keep the original structure and formatting
5. Add ALL relevant links from the collection
6. If a link is mentioned but missing from collection, keep the reference without the link"""

            # Get response using get_response() from BaseAgent
            response = await self.agent.get_response(
                prompt.format(draft=draft, links=formatted_links)
            )
            
            state.results[self.id] = {
                "response": response,
                "links_added": len(links)
            }
            
            return "format"
            
        except Exception as e:
            logger.error(f"Error enriching content: {str(e)}")
            return None

class AnalysisNode(LLMNode):
    """Node that analyzes collected content."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        try:
            collected_content = state.results.get("collect", {}).get("content", [])
            logger.info(f"Analyzing {len(collected_content)} content items")
            
            # Group content by type/source for better analysis
            grouped_content = self._group_content(collected_content)
            
            # Create detailed analysis prompt
            analysis_prompt = ANALYSIS_PROMPT.format(
                collect={
                    "content": grouped_content,
                    "date_range": state.results["collect"]["date_range"]
                }
            )
            
            response = await self.agent.get_response(analysis_prompt)
            
            # Store analysis results with content mapping
            state.results[self.id] = {
                "response": response,
                "content_analyzed": len(collected_content),
                "content_groups": grouped_content
            }
            
            logger.info(f"Analysis complete. Content groups: {list(grouped_content.keys())}")
            return "draft"
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return None
    
    def _group_content(self, content: List[Dict]) -> Dict:
        """Group content by type for better analysis."""
        grouped = {
            "discussions": [],
            "shared_links": [],
            "announcements": [],
            "technical_content": []
        }
        
        for item in content:
            # Categorize based on content characteristics
            if item.get("urls"):
                grouped["shared_links"].append(item)
            if "announcement" in item.get("clean_content", "").lower():
                grouped["announcements"].append(item)
            if any(tech in item.get("clean_content", "").lower() 
                  for tech in ["code", "api", "model", "implementation"]):
                grouped["technical_content"].append(item)
            # Add to discussions if it doesn't fit other categories
            if not any(item in group for group in grouped.values()):
                grouped["discussions"].append(item)
        
        return grouped

class DraftNode(LLMNode):
    """Node that drafts newsletter content."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        try:
            analysis = state.results.get("analyze", {})
            if not analysis:
                logger.warning("No analysis available for draft")
                return None
            
            # Include content grouping in draft prompt
            draft_prompt = DRAFT_PROMPT.format(
                analyze={
                    "response": analysis["response"],
                    "content_groups": analysis.get("content_groups", {}),
                    "total_analyzed": analysis.get("content_analyzed", 0)
                }
            )
            
            response = await self.agent.get_response(draft_prompt)
            
            state.results[self.id] = {
                "response": response,
                "based_on_content": analysis.get("content_analyzed", 0)
            }
            
            logger.info(f"Draft generated based on {analysis.get('content_analyzed', 0)} content items")
            return "enrich"
            
        except Exception as e:
            logger.error(f"Error in draft: {str(e)}")
            return None

async def create_newsletter_workflow(*, additional_context: str = "") -> Graph:
    """
    Create newsletter generation workflow.
    
    Args:
        additional_context (str): Any additional context or instructions for this newsletter run
    """
    logger.info("Creating newsletter workflow...")
    
    # Create agent
    agent = BaseAgent(provider="openpipe", persona=PERSONA)
    
    # Create nodes
    collect = ContentCollectorNode(
        id="collect",
        agent=agent,
        next_nodes={
            "analyze": "analyze"
        }
    )
    
    analyze = LLMNode(
        id="analyze",
        agent=agent,
        prompt=ANALYSIS_PROMPT,
        next_nodes={"default": "draft"}
    )
    
    draft = LLMNode(
        id="draft",
        agent=agent,
        prompt=DRAFT_PROMPT,
        next_nodes={"default": "enrich"}
    )
    
    enrich = LinkEnricherNode(
        id="enrich",
        agent=agent,
        next_nodes={
            "default": "format"
        }
    )
    
    format_node = NewsletterFormatterNode(
        id="format",
        agent=agent,
        next_nodes={},  # Empty dict for terminal node
        additional_context=additional_context
    )
    
    # Create and validate graph
    graph = Graph()
    graph.add_node(collect)
    graph.add_node(analyze)
    graph.add_node(draft)
    graph.add_node(enrich)
    graph.add_node(format_node)
    
    # Add entry point
    graph.add_entry_point("main", "collect")
    
    logger.info("Validating newsletter workflow graph...")
    graph.validate()
    
    return graph

async def main(additional_context: str = ""):
    """
    Initialize and run the newsletter workflow.
    
    Args:
        additional_context (str): Any additional context or instructions for this newsletter run
    """
    try:
        # Load environment
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Use reader-specific tokens
        reader_token = os.getenv("DISCORD_READER_TOKEN")
        reader_channel_id = os.getenv("DISCORD_READER_CHANNEL_ID")
        
        if not reader_token or not reader_channel_id:
            raise ValueError("Missing Discord configuration")
        
        # Configure runtime
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        
        config = RuntimeConfig(
            provider="openpipe",
            persona=PERSONA,
            platform_config={
                "intents": intents,
                "activity_type": "watching",
                "activity_name": "for newsletter content"
            }
        )
        
        # Create workflow with additional context
        workflow = await create_newsletter_workflow(additional_context=additional_context)
        
        # Initialize Discord client directly
        client = discord.Client(intents=intents)
        await client.login(reader_token)
        
        # Fetch messages without full client connection
        channel = await client.fetch_channel(int(reader_channel_id))
        messages = []
        async for message in channel.history(limit=MESSAGES_TO_FETCH, oldest_first=False):
            # Extract URLs from message content and embeds
            urls = []
            
            # Get URLs from embeds
            for embed in message.embeds:
                if embed.url:
                    urls.append(embed.url)
            
            # Add message data with extracted URLs
            messages.append({
                "content": message.content,
                "clean_content": message.clean_content,
                "embeds": [
                    {
                        "url": e.url,
                        "title": e.title,
                        "description": e.description,
                        "image": e.image.url if e.image else None,
                        "fields": [{"name": f.name, "value": f.value} for f in e.fields]
                    }
                    for e in message.embeds
                ],
                "urls": urls,
                "attachments": [
                    {
                        "filename": a.filename,
                        "url": a.url,
                        "content_type": a.content_type
                    }
                    for a in message.attachments
                ],
                "timestamp": message.created_at.timestamp(),
                "author": {
                    "name": message.author.name,
                    "id": message.author.id
                },
                "edited_timestamp": message.edited_at.timestamp() if message.edited_at else None
            })
        
        # Close client connection
        await client.close()
        
        if messages:
            # Run workflow
            state = NodeState(
                context=NodeContext(
                    metadata={"recent_messages": messages}
                )
            )
            
            final_state = await workflow.run("main", state)
            
            # Print results
            print("\n📊 PIPELINE RESULTS")
            print("="*80)
            
            if "collect" in final_state.results:
                print("\n🔍 COLLECTION PHASE")
                print("-"*40)
                print(f"Messages processed: {final_state.results['collect']['total_messages']}")
                print(f"Date range: {final_state.results['collect']['date_range']}")
            
            if "format" in final_state.results:
                print("\n🔄 FINAL NEWSLETTER")
                print("="*80)
                print(final_state.results["format"]["response"])
            
        else:
            logger.warning("No messages found in channel history")
        
    except Exception as e:
        logger.error(f"Error running newsletter workflow: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Reduce noise from Discord
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Example of how to use additional context
    first_edition_context = """
    Special Instructions:
    - This is the first edition of the Improbable Automata newsletter
    - We should introduce some things like:
        - The newsletter's purpose: to be a leading example of how to build an AI first business and use AI agents and other automations to build products and 10x your operations
        - What Improbable Automata means and why this newsletter is different: we are building an ai agent personality into a newsletter that will grow and shape with its audience. Our name takes its inspiration from two ideas: cellular automata and the infinite improbability drive. Cellular automata is the idea that agent systems can evolve and change over time. the infinite improbability drive represnts that idea that if we simply allow such systems to exist, they will become much more than the sum of their parts. Perhaps taking on a life of its own.
        - The current state of autonomy of the newsletter: we built the AI system behind it code named 'alchemist' (we don't talk much about this publicly its not released yet). alchemist is a prototype agent system that allows us to configure the specific mind design of the newsletter system. we are simultaneously working on various other minds and ai systems. Right now, after we generate the newsletter using our AI agent system, we add a human touch. The plan is to run it this way for a while until we get to a more refined style, then do some training and have it run autonomously.
    - The overarching thematic aura of the first edition is that we are giving a bit of an intro to the newsletter and a bit of a year in review of 2024 in AI.
    """
    
    asyncio.run(main(additional_context=first_edition_context))