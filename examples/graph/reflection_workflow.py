"""
Reflection Workflow Example: Multi-step reasoning with Chain of Thought.
"""

import asyncio
from datetime import datetime
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.graph.base import Graph
from alchemist.ai.graph.nodes.agent import AgentNode
from alchemist.ai.graph.nodes.terminal import TerminalNode
from alchemist.ai.graph.state import NodeState
from alchemist.ai.base.logging import (
    configure_logging, 
    LogLevel, 
    LogComponent, 
    get_logger,
    Colors,
    VerbosityLevel,
    AlchemistLoggingConfig
)

# Update logging configuration to show less debug noise
configure_logging(
    default_level=LogLevel.INFO,
    component_levels={
        LogComponent.WORKFLOW: LogLevel.INFO,
        LogComponent.AGENT: LogLevel.INFO,
        LogComponent.GRAPH: LogLevel.INFO,
        LogComponent.NODES: LogLevel.INFO
    }
)

logger = get_logger(LogComponent.WORKFLOW)

async def run_reflection_workflow() -> None:
    """
    Build and execute the reflection workflow.
    """
    graph = Graph(
        logging_config=AlchemistLoggingConfig(
            level=VerbosityLevel.INFO,
            show_llm_messages=True,
            show_node_transitions=True,
            show_tool_calls=True
        )
    )
    
    agent = BaseAgent()
    
    # Define reflection nodes with detailed logging callback
    def log_thinking_step(step_name: str):
        async def callback(state: NodeState, node_id: str) -> None:
            if node_id in state.results:
                result = state.results[node_id]
                print(f"\n{Colors.BOLD}{'=' * 50}{Colors.RESET}")
                print(f"{Colors.BOLD}🤔 {step_name}{Colors.RESET}")
                print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}")
                print(f"{Colors.INFO}{result.get('response', '')}{Colors.RESET}")
                print(f"{Colors.DIM}Time: {result.get('timing', 0):.1f}s{Colors.RESET}")
                print(f"{Colors.BOLD}{'=' * 50}{Colors.RESET}\n")
                
                # Also log to logger for file output
                logger.info(
                    f"\n💭 {step_name} completed in {result.get('timing', 0):.1f}s"
                )
        return callback

    initial_reflection = AgentNode(
        id="initial_reflection",
        prompt="Step 1: Initial Assessment\nThe user said: {user_input}\nProvide your initial thoughts and assessment.",
        agent=agent,
        input_map={"user_input": "data.input_text"},
        next_nodes={"default": "deep_reflection"},
        metadata={"on_complete": log_thinking_step("Initial Assessment")}
    )

    deep_reflection = AgentNode(
        id="deep_reflection",
        prompt="Step 2: Deep Analysis\nInitial assessment: {initial_thoughts}\nNow analyze deeper implications.",
        agent=agent,
        input_map={"initial_thoughts": "node.initial_reflection.response"},
        next_nodes={"default": "final_synthesis"},
        metadata={"on_complete": log_thinking_step("Deep Analysis")}
    )

    final_synthesis = AgentNode(
        id="final_synthesis",
        prompt="Step 3: Final Synthesis\nInitial: {initial_thoughts}\nDeep: {deep_analysis}\nSynthesize a final response.",
        agent=agent,
        input_map={
            "initial_thoughts": "node.initial_reflection.response",
            "deep_analysis": "node.deep_reflection.response"
        },
        next_nodes={"default": "end"},
        metadata={"on_complete": log_thinking_step("Final Synthesis")}
    )

    # Add nodes and set entry point
    for node in [initial_reflection, deep_reflection, final_synthesis, TerminalNode(id="end")]:
        graph.add_node(node)
    graph.add_entry_point("start", "initial_reflection")

    # Run workflow with better progress indication
    state = NodeState()
    state.set_data(
        "input_text", 
        "If doug's dad is bob's cousin, and bob's dad is dave's dad, what is the relationship between doug and dave?"
    )
    
    print(f"\n{Colors.BOLD}🔍 Analyzing Question:{Colors.RESET}")
    print(f"{Colors.INFO}{state.data['input_text']}{Colors.RESET}\n")
    
    start_time = datetime.now()
    final_state = await graph.run("start", state)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{Colors.SUCCESS}✨ Analysis Complete in {elapsed:.1f}s{Colors.RESET}\n")

if __name__ == "__main__":
    asyncio.run(run_reflection_workflow())