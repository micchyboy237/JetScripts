from agents import Agent, Runner, function_tool
from agents import Agent, Runner, handoffs, function_tool
from jet.logger import CustomLogger
from mem0 import MemoryClient
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: MLX Agents SDK
---

Integrate [**Mem0**](https://github.com/mem0ai/mem0) with [MLX Agents SDK](https://github.com/openai/openai-agents-python), a lightweight framework for building multi-agent workflows. This integration enables agents to access persistent memory across conversations, enhancing context retention and personalization.

## Overview

1. Store and retrieve memories from Mem0 within MLX agents
2. Multi-agent workflows with shared memory
3. Retrieve relevant memories for past conversations
4. Personalized responses based on user history

## Prerequisites

Before setting up Mem0 with MLX Agents SDK, ensure you have:

1. Installed the required packages:
"""
logger.info("## Overview")

pip install openai-agents mem0ai

"""
2. Valid API keys:
   - [Mem0 API Key](https://app.mem0.ai/dashboard/api-keys)
   - [MLX API Key](https://platform.openai.com/api-keys)

## Basic Integration Example

The following example demonstrates how to create an MLX agent with Mem0 memory integration:
"""
logger.info("## Basic Integration Example")


# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["MEM0_API_KEY"] = "your-mem0-api-key"

mem0 = MemoryClient()

@function_tool
def search_memory(query: str, user_id: str) -> str:
    """Search through past conversations and memories"""
    memories = mem0.search(query, user_id=user_id, limit=3)
    if memories and memories.get('results'):
        return "\n".join([f"- {mem['memory']}" for mem in memories['results']])
    return "No relevant memories found."

@function_tool
def save_memory(content: str, user_id: str) -> str:
    """Save important information to memory"""
    mem0.add([{"role": "user", "content": content}], user_id=user_id)
    return "Information saved to memory."

agent = Agent(
    name="Personal Assistant",
    instructions="""You are a helpful personal assistant with memory capabilities.
    Use the search_memory tool to recall past conversations and user preferences.
    Use the save_memory tool to store important information about the user.
    Always personalize your responses based on available memory.""",
    tools=[search_memory, save_memory],
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats"
)

def chat_with_agent(user_input: str, user_id: str) -> str:
    """
    Handle user input with automatic memory integration.

    Args:
        user_input: The user's message
        user_id: Unique identifier for the user

    Returns:
        The agent's response
    """
    result = Runner.run_sync(agent, user_input)

    return result.final_output

if __name__ == "__main__":

    response_1 = chat_with_agent(
        "I love Italian food and I'm planning a trip to Rome next month",
        user_id="alice"
    )
    logger.debug(response_1)

    response_2 = chat_with_agent(
        "Give me some recommendations for food",
        user_id="alice"
    )
    logger.debug(response_2)

"""
## Multi-Agent Workflow with Handoffs

Create multiple specialized agents with proper handoffs and shared memory:
"""
logger.info("## Multi-Agent Workflow with Handoffs")


travel_agent = Agent(
    name="Travel Planner",
    instructions="""You are a travel planning specialist. Use get_user_context to
    understand the user's travel preferences and history before making recommendations.
    After providing your response, use store_conversation to save important details.""",
    tools=[search_memory, save_memory],
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats"
)

health_agent = Agent(
    name="Health Advisor",
    instructions="""You are a health and wellness advisor. Use get_user_context to
    understand the user's health goals and dietary preferences.
    After providing advice, use store_conversation to save relevant information.""",
    tools=[search_memory, save_memory],
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats"
)

triage_agent = Agent(
    name="Personal Assistant",
    instructions="""You are a helpful personal assistant that routes requests to specialists.
    For travel-related questions (trips, hotels, flights, destinations), hand off to Travel Planner.
    For health-related questions (fitness, diet, wellness, exercise), hand off to Health Advisor.
    For general questions, you can handle them directly using available tools.""",
    handoffs=[travel_agent, health_agent],
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats"
)

def chat_with_handoffs(user_input: str, user_id: str) -> str:
    """
    Handle user input with automatic agent handoffs and memory integration.

    Args:
        user_input: The user's message
        user_id: Unique identifier for the user

    Returns:
        The agent's response
    """
    result = Runner.run_sync(triage_agent, user_input)

    conversation = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": result.final_output}
    ]
    mem0.add(conversation, user_id=user_id)

    return result.final_output

response = chat_with_handoffs("Plan a healthy meal for my Italy trip", user_id="alex")
logger.debug(response)

"""
## Quick Start Chat Interface

Simple interactive chat with memory:
"""
logger.info("## Quick Start Chat Interface")

def interactive_chat():
    """Interactive chat interface with memory and handoffs"""
    user_id = input("Enter your user ID: ") or "demo_user"
    logger.debug(f"Chat started for user: {user_id}")
    logger.debug("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = chat_with_handoffs(user_input, user_id)
        logger.debug(f"Assistant: {response}\n")

if __name__ == "__main__":
    interactive_chat()

"""
## Key Features

### 1. Automatic Memory Integration
- **Tool-Based Memory**: Agents use function tools to search and save memories
- **Conversation Storage**: All interactions are automatically stored
- **Context Retrieval**: Agents can access relevant past conversations

### 2. Multi-Agent Memory Sharing
- **Shared Context**: Multiple agents access the same memory store
- **Specialized Agents**: Create domain-specific agents with shared memory
- **Seamless Handoffs**: Agents maintain context across handoffs

### 3. Flexible Memory Operations
- **Retrieve Capabilities**: Retrieve relevant memories from previous conversation
- **User Segmentation**: Organize memories by user ID
- **Memory Management**: Built-in tools for saving and retrieving information

## Configuration Options

Customize memory behavior:
"""
logger.info("## Key Features")

memories = mem0.search(
    query="travel preferences",
    user_id="alex",
    limit=5  # Number of memories to retrieve
)

mem0.add(
    messages=[{"role": "user", "content": "I prefer luxury hotels"}],
    user_id="alex",
    metadata={"category": "travel", "importance": "high"}
)

"""
## Help

- [MLX Agents SDK Documentation](https://openai.github.io/openai-agents-python/)
- [Mem0 Platform](https://app.mem0.ai/)
- If you need further assistance, please feel free to reach out to us through the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Help")

logger.info("\n\n[DONE]", bright=True)