"""Tests for the calculator tool."""

import pytest
from mirascope.core import BaseMessageParam, openai
from alchemist.ai.tools.calculator import CalculatorTool

def test_tool_metadata():
    """Test tool name and description."""
    tool = CalculatorTool(expression="2 + 2")
    assert tool._name() == "calculate"
    assert "mathematical" in tool._description().lower()
    assert "expression" in tool.model_fields

def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    cases = [
        ("2 + 2", "4"),
        ("10 - 5", "5"),
        ("3 * 4", "12"),
        ("15 / 3", "5.0"),
    ]
    
    for expression, expected in cases:
        tool = CalculatorTool(expression=expression)
        assert tool.call() == expected

def test_complex_expressions():
    """Test more complex mathematical expressions."""
    tool = CalculatorTool(expression="2 ** 3")
    assert tool.call() == "8"
    
    tool = CalculatorTool(expression="(4 + 2) * 3")
    assert tool.call() == "18"
    
    tool = CalculatorTool(expression="42 ** 0.5")  # Square root
    assert float(tool.call()) == pytest.approx(6.4807, rel=1e-4)

def test_error_handling():
    """Test error handling for invalid expressions."""
    cases = [
        "2 + ",  # Incomplete
        "invalid",  # Invalid name
        "1 / 0",  # Division by zero
        "print('hack')",  # Unsafe operation
    ]
    
    for expression in cases:
        tool = CalculatorTool(expression=expression)
        result = tool.call()
        assert result.startswith("Error:")
        assert isinstance(result, str) 