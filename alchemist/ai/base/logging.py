"""Logging Configuration

This module provides centralized logging configuration for the Alchemist system.
Allows easy toggling of different logging levels for different components.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum
import json
from datetime import datetime
from pathlib import Path

# Log format presets
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"

class LogFormat(str, Enum):
    """Predefined log formats."""
    DEFAULT = DEFAULT_FORMAT
    DEBUG = DEBUG_FORMAT
    SIMPLE = SIMPLE_FORMAT

class LogComponent(str, Enum):
    """Components that can be logged.
    
    Each component represents a major subsystem in the Alchemist framework.
    The value is the logger name used in the logging hierarchy.
    """
    AGENT = "alchemist.ai.base.agent"          # Base agent functionality
    RUNTIME = "alchemist.ai.base.runtime"       # Runtime environment
    GRAPH = "alchemist.ai.graph.base"          # Graph system core
    NODES = "alchemist.ai.graph.nodes"         # Graph nodes
    TOOLS = "alchemist.ai.tools"               # Tool implementations
    PROMPTS = "alchemist.ai.prompts"           # Prompt management
    DISCORD = "alchemist.core.extensions.discord"  # Discord integration
    WORKFLOW = "alchemist.ai.graph.workflow"    # Workflow execution
    SESSION = "alchemist.core.session"          # Session management
    MEMORY = "alchemist.core.memory"           # Memory systems

class LogLevel(int, Enum):
    """Log levels mapped to logging module levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

def configure_logging(
    default_level: LogLevel = LogLevel.INFO,
    component_levels: Optional[Dict[LogComponent, LogLevel]] = None,
    format_string: str = DEFAULT_FORMAT,
    log_file: Optional[str] = None,
    enable_json: bool = False
) -> None:
    """Configure logging for all components.
    
    Args:
        default_level: Default logging level for all components
        component_levels: Optional dict of component-specific levels
        format_string: Format string for log messages
        log_file: Optional file path to write logs to
        enable_json: Whether to enable JSON formatting for logs
    """
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(default_level.value)
    
    # Remove existing handlers and add new ones
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Configure component-specific levels
    if component_levels:
        for component, level in component_levels.items():
            logger = logging.getLogger(component.value)
            logger.setLevel(level.value)
            if enable_json:
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                json_handler = JsonLogHandler()
                json_handler.setFormatter(logging.Formatter(format_string))
                logger.addHandler(json_handler)

class JsonLogHandler(logging.Handler):
    """Handler that formats log records as JSON."""
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as JSON."""
        try:
            msg = self.format(record)
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
                "function": record.funcName,
                "line": record.lineno,
                "path": record.pathname
            }
            
            # Add extra fields if present
            if hasattr(record, "extra"):
                log_entry.update(record.extra)
            
            print(json.dumps(log_entry))
            
        except Exception as e:
            print(f"Error in JSON logging: {e}")

def get_logger(component: LogComponent) -> logging.Logger:
    """Get a logger for a specific component.
    
    Args:
        component: The component to get a logger for
        
    Returns:
        Logger configured for the component
    """
    return logging.getLogger(component.value)

def log_state(logger: logging.Logger, state: Dict[str, Any], prefix: str = "") -> None:
    """Log a state dictionary in a readable format.
    
    Args:
        logger: Logger to use
        state: State dictionary to log
        prefix: Optional prefix for log messages
    """
    try:
        for key, value in state.items():
            if isinstance(value, dict):
                logger.debug(f"{prefix}{key}:")
                log_state(logger, value, prefix + "  ")
            else:
                logger.debug(f"{prefix}{key}: {value}")
    except Exception as e:
        logger.error(f"Error logging state: {e}")

def set_component_level(component: LogComponent, level: LogLevel) -> None:
    """Set logging level for a specific component.
    
    Args:
        component: Component to configure
        level: Logging level to set
    """
    logging.getLogger(component.value).setLevel(level.value)

def disable_all_logging() -> None:
    """Disable logging for all components."""
    for component in LogComponent:
        logging.getLogger(component.value).setLevel(logging.CRITICAL)

def enable_debug_mode() -> None:
    """Enable debug logging for all components."""
    for component in LogComponent:
        logging.getLogger(component.value).setLevel(logging.DEBUG) 