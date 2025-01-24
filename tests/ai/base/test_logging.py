"""Tests for the logging system."""

import pytest
import logging
from pathlib import Path
import json
import os
from typing import Dict, Any

from alchemist.ai.base.logging import (
    LogComponent,
    LogLevel,
    LogFormat,
    configure_logging,
    get_logger,
    log_state,
    JsonLogHandler
)

@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file."""
    return str(tmp_path / "test.log")

def test_basic_logging(temp_log_file):
    """Test basic logging configuration."""
    # Configure logging
    configure_logging(
        default_level=LogLevel.DEBUG,
        format_string=LogFormat.DEFAULT,
        log_file=temp_log_file
    )
    
    # Get a test logger
    logger = get_logger(LogComponent.AGENT)
    
    # Log some messages
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    
    # Verify log file exists and contains messages
    assert Path(temp_log_file).exists()
    with open(temp_log_file) as f:
        content = f.read()
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content

def test_component_levels(temp_log_file):
    """Test component-specific log levels."""
    # Configure with different levels per component
    configure_logging(
        default_level=LogLevel.INFO,
        component_levels={
            LogComponent.AGENT: LogLevel.DEBUG,
            LogComponent.GRAPH: LogLevel.WARNING
        },
        log_file=temp_log_file
    )
    
    # Get loggers
    agent_logger = get_logger(LogComponent.AGENT)
    graph_logger = get_logger(LogComponent.GRAPH)
    
    # Log messages at different levels
    agent_logger.debug("Agent debug")  # Should appear
    graph_logger.debug("Graph debug")  # Should not appear
    agent_logger.warning("Agent warning")  # Should appear
    graph_logger.warning("Graph warning")  # Should appear
    
    # Verify log content
    with open(temp_log_file) as f:
        content = f.read()
        assert "Agent debug" in content
        assert "Graph debug" not in content
        assert "Agent warning" in content
        assert "Graph warning" in content

def test_json_logging(capsys):
    """Test JSON log formatting."""
    # Configure with JSON logging
    configure_logging(
        default_level=LogLevel.DEBUG,
        enable_json=True
    )
    
    # Get a test logger
    logger = get_logger(LogComponent.AGENT)
    
    # Log a test message
    test_msg = "Test JSON logging"
    logger.info(test_msg)
    
    # Capture output and verify JSON format
    captured = capsys.readouterr()
    log_entry = json.loads(captured.out)
    
    assert log_entry["message"] == test_msg
    assert log_entry["level"] == "INFO"
    assert log_entry["logger"] == LogComponent.AGENT.value

def test_log_state():
    """Test state dictionary logging."""
    # Configure logging
    configure_logging(default_level=LogLevel.DEBUG)
    
    # Create test state
    test_state: Dict[str, Any] = {
        "level1": {
            "level2": {
                "key": "value"
            },
            "number": 42
        },
        "string": "test"
    }
    
    # Log state
    logger = get_logger(LogComponent.GRAPH)
    log_state(logger, test_state)
    
    # Since we can't easily capture debug output, we just verify no errors occur
    # In real usage, you would see the formatted output in the logs

def test_log_formats():
    """Test different log formats."""
    for format_enum in LogFormat:
        configure_logging(format_string=format_enum.value)
        logger = get_logger(LogComponent.AGENT)
        logger.info("Test message")  # Visual verification of format

if __name__ == "__main__":
    pytest.main([__file__]) 