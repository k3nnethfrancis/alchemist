"""
Core logging functionality for agents.

Provides decorators and utilities for logging agent conversations and interactions.
Currently focused on end-of-session logging with plans to expand to context window 
management and other features.
"""

import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

def log_session(log_dir: str = "logs/sessions") -> Callable:
    """
    Decorator that logs an agent's conversation history at the end of a session.
    
    Args:
        log_dir: Directory to store session logs. Defaults to "logs/sessions"
        
    Returns:
        Callable: Decorated function that handles logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(agent: Any, *args, **kwargs) -> Any:
            # Create session ID and log path
            session_id = str(uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(log_dir) / f"{timestamp}_{session_id}.json"
            
            # Ensure log directory exists
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run the original function
            result = func(agent, *args, **kwargs)
            
            # Log the conversation history
            history = []
            for msg in agent.history:
                if hasattr(msg, 'model_dump'):
                    history.append(msg.model_dump())
                else:
                    history.append(msg)
            
            # Write to file
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'session_id': session_id,
                    'timestamp': timestamp,
                    'provider': agent.provider,
                    'history': history
                }, f, indent=2, ensure_ascii=False)
            
            return result
        return wrapper
    return decorator