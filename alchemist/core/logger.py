"""
Core logging functionality for agents.

Provides decorators and utilities for logging agent conversations and interactions.
"""

import json
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

def log_step(log_dir: str = "logs/chat") -> Callable:
    """
    Decorator for logging individual conversation steps.
    
    Args:
        log_dir: Directory to store chat logs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(runtime: Any, message: str, *args, **kwargs) -> Any:
            # Get response first
            response = await func(runtime, message, *args, **kwargs)
            
            if runtime.current_session:
                session = runtime.current_session
                log_path = Path(log_dir) / f"{session.id}.json"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing log if it exists
                if log_path.exists():
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                else:
                    log_data = {
                        'session_id': session.id,
                        'timestamp': session.start_time.isoformat(),
                        'platform': session.platform,
                        'provider': runtime.agent.provider,
                        'messages': []
                    }
                
                # Add new message pair
                log_data['messages'].append({
                    'timestamp': datetime.now().isoformat(),
                    'user': message,
                    'assistant': response
                })
                
                # Write updated log
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            return response
        return wrapper
    return decorator

def log_run(log_dir: str = "logs/chat") -> Callable:
    """
    Decorator for logging entire chat sessions.
    
    Args:
        log_dir: Directory to store chat logs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(agent: Any, *args, **kwargs) -> Any:
            session_id = str(uuid4())
            log_path = Path(log_dir) / f"session_{session_id}.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize session log
            log_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'provider': agent.provider,
                'messages': []
            }
            
            # Write initial log
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            # Run the function
            return await func(agent, *args, **kwargs)
            
        return wrapper
    return decorator