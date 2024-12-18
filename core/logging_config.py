"""
Logging configuration for the Discord bot.

This module implements structured logging to track agent behavior, Discord interactions,
and system events. It provides detailed insights into the bot's operation and
decision-making process through multiple logging handlers and formatted output.

Example:
    logger = BotLogger()
    logger.log_agent_decision({
        "event": "message_received",
        "channel_id": "123456789",
        "content_length": 50
    })
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any
import os

class BotLogger:
    """
    Custom logger for tracking bot interactions and behavior.
    
    Provides structured logging with both console and file output for:
    - Agent decisions and responses
    - Discord message handling
    - Memory operations
    - Model interactions
    
    Attributes:
        logger: Base logging instance
        log_dir: Directory for log files
    """

    def __init__(self):
        self.logger = logging.getLogger('ElizaBot')
        self.log_dir = 'logs'
        self._setup_logger()

    def _setup_logger(self) -> None:
        """
        Configure logging format and handlers.
        
        Sets up:
        - Console handler with colored output
        - File handler for detailed logs
        - JSON handler for structured logging
        """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler(
            f"{self.log_dir}/eliza_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(file_formatter)

        # Set up logger
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log_agent_decision(self, context: Dict[str, Any]) -> None:
        """
        Log agent's decision-making process with structured data.
        
        Args:
            context: Dictionary containing decision context including:
                - event: Type of event being logged
                - Additional relevant metadata
        
        Example:
            log_agent_decision({
                "event": "response_generated",
                "response_length": 150,
                "processing_time": 0.5
            })
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent_decision",
            **context
        }
        self.logger.info(json.dumps(log_entry))