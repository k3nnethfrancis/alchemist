"""Test OpenPipe provider implementation."""

from openpipe import OpenAI
from pprint import pprint
from mirascope.core.base import BaseMessageParam, BaseDynamicConfig, BaseTool
from pydantic import Field
from core.providers.openpipe import openpipe_call
import asyncio
from contextlib import asynccontextmanager

# Test tool for our implementation
class EchoTool(BaseTool):
    """Simple echo tool for testing."""
    
    message: str = Field(..., description="Message to echo back")
    
    @classmethod
    def tool_schema(cls) -> dict:
        """Tool schema for LLM."""
        return {
            "name": "echo",
            "description": "Echoes back the message",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
                "required": ["message"]
            }
        }
    
    async def call(self) -> dict:
        """Echo the message."""
        return {"echoed": self.message}

@asynccontextmanager
async def get_client():
    """Get OpenPipe client in async context."""
    client = OpenAI()
    try:
        yield client
    finally:
        await client.close()  # Properly close client

async def test_raw_openpipe():
    """Test raw OpenPipe API to verify connection."""
    async with get_client() as client:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            metadata={"prompt_id": "test", "provider": "openpipe"},
            store=True
        )
        return response

def test_raw_openpipe_tool():
    """Test raw OpenPipe API with tool to verify response format."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant. When asked to echo something, use the echo tool."
            },
            {
                "role": "user", 
                "content": "Echo this message: 'Hello World'"
            }
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echoes back the message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            }
        }],
        tool_choice="auto",
        metadata={"prompt_id": "test_tool", "provider": "openpipe"},
        store=True
    )
    return response

@openpipe_call("gpt-4o-mini")
def test_provider_chat(query: str) -> BaseDynamicConfig:
    """Test our OpenPipe provider implementation."""
    messages = [
        BaseMessageParam(role="system", content="You are a helpful assistant."),
        BaseMessageParam(role="user", content=query)
    ]
    return {"messages": messages}

@openpipe_call("gpt-4o-mini", tools=[EchoTool])
def test_provider_tool(query: str) -> BaseDynamicConfig:
    """Test our OpenPipe tool implementation."""
    messages = [
        BaseMessageParam(
            role="system", 
            content=(
                "You are a helpful assistant. When asked to echo something, "
                "ALWAYS use the echo tool to respond. Do not explain what you're doing, "
                "just use the tool."
            )
        ),
        BaseMessageParam(role="user", content=query)
    ]
    return {"messages": messages}

async def main():
    """Run test suite."""
    try:
        # Test 1: Raw OpenPipe API
        print("\nTest 1: Raw OpenPipe API")
        print("-" * 30)
        raw_response = await test_raw_openpipe()
        pprint(raw_response.model_dump())
        
        # Test 2: Our Provider Implementation
        print("\nTest 2: Custom Provider")
        print("-" * 30)
        response = test_provider_chat("Hello, how are you?")
        
        print("Response Content:", response.content)
        print("Model:", response.model)
        print("Usage:", response.usage)
        print("ID:", response.id)
        
        # Test 3: Compare Responses
        print("\nTest 3: Response Comparison")
        print("-" * 30)
        print("Raw Response ID:", raw_response.id)
        print("Provider Response ID:", response.id)
        print("Fields Match:", raw_response.id == response.id)
        
        # Test 4: Tool Testing
        print("\nTest 4: Tool Testing")
        print("-" * 30)
        tool_response = test_provider_tool("Echo this message: 'Hello World'")
        
        print("Response Content:", tool_response.content)
        print("Has Tool:", bool(tool_response.tool))
        
        if tool := tool_response.tool:
            print("\nTool detected!")
            print("Tool Type:", type(tool).__name__)
            print("Tool Args:", tool.model_dump())
            result = await tool.call()
            print("Tool Result:", result)
            
            # Test tool message params
            tool_messages = tool_response.tool_message_params([(tool, result)])
            print("\nTool Messages:")
            for msg in tool_messages:
                print(f"Role: {msg['role']}")  # Dictionary access
                print(f"Content: {msg['content']}")  # Dictionary access
                print(f"Available fields:", msg.keys())  # Dictionary method
                if 'tool_calls' in msg:
                    print(f"Tool Calls: {msg['tool_calls']}")
                print("-" * 15)
        
        # Test 5: Raw Tool API
        print("\nTest 5: Raw Tool API")
        print("-" * 30)
        raw_tool_response = test_raw_openpipe_tool()
        print("Raw Tool Response:")
        pprint(raw_tool_response.model_dump())
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
