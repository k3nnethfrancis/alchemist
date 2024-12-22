"""
Core logging functionality for agents.

Provides decorators and utilities for logging agent conversations and interactions
through different interfaces (CLI, Discord, etc.).
"""

import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Union
from uuid import uuid4

def log_run(log_dir: str = "logs/chat") -> Callable:
    """
    Decorator for logging complete agent runs (CLI mode).
    
    Args:
        log_dir: Directory to store complete session logs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(agent: Any, *args, **kwargs) -> Any:
            session_id = str(uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(log_dir) / f"{timestamp}_{session_id}.json"
            
            result = await func(agent, *args, **kwargs)
            
            # Log complete conversation history
            history = []
            for msg in agent.history:
                if hasattr(msg, 'model_dump'):
                    history.append(msg.model_dump())
                else:
                    history.append(msg)
                    
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'session_id': session_id,
                    'timestamp': timestamp,
                    'provider': agent.provider,
                    'interface': 'cli',
                    'history': history
                }, f, indent=2, ensure_ascii=False)
                
            return result
        return wrapper
    return decorator

def log_step(log_dir: str = "data/sessions") -> Callable:
    """
    Decorator for logging individual conversation steps (Discord, etc.).
    
    Args:
        log_dir: Directory to store step-by-step logs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(runtime: Any, message: str, *args, **kwargs) -> Any:
            if runtime.current_session:
                session = runtime.current_session
                log_path = Path(log_dir) / session.interface / f"{session.session_id}.json"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing log if it exists
                if log_path.exists():
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                else:
                    log_data = {
                        'session_id': session.session_id,
                        'timestamp': session.start_time.isoformat(),
                        'provider': runtime.agent.provider,
                        'interface': session.interface,
                        'messages': []
                    }
                
                # Add new message
                log_data['messages'].extend(session.messages[-2:])  # Add latest user+assistant pair
                
                # Write updated log
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            return await func(runtime, message, *args, **kwargs)
        return wrapper
    return decorator