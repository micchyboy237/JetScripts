from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
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
title: Google Agent Development Kit
---

Integrate [**Mem0**](https://github.com/mem0ai/mem0) with [Google Agent Development Kit (ADK)](https://github.com/google/adk-python), an open-source framework for building multi-agent workflows. This integration enables agents to access persistent memory across conversations, enhancing context retention and personalization.

## Overview

1. Store and retrieve memories from Mem0 within Google ADK agents
2. Multi-agent workflows with shared memory across hierarchies
3. Retrieve relevant memories from past conversations
4. Personalized responses

## Prerequisites

Before setting up Mem0 with Google ADK, ensure you have:

1. Installed the required packages:
"""
logger.info("## Overview")

pip install google-adk mem0ai

"""
2. Valid API keys:
   - [Mem0 API Key](https://app.mem0.ai/dashboard/api-keys)
   - Google AI Studio API Key

## Basic Integration Example

The following example demonstrates how to create a Google ADK agent with Mem0 memory integration:
"""
logger.info("## Basic Integration Example")


os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
os.environ["MEM0_API_KEY"] = "your-mem0-api-key"

mem0 = MemoryClient()

def search_memory(query: str, user_id: str) -> dict:
    """Search through past conversations and memories"""
    memories = mem0.search(query, user_id=user_id)
    if memories.get('results', []):
        memory_context = "\n".join([f"- {mem['memory']}" for mem in memories.get('results', [])])
        return {"status": "success", "memories": memory_context}
    return {"status": "no_memories", "message": "No relevant memories found"}

def save_memory(content: str, user_id: str) -> dict:
    """Save important information to memory"""
    try:
        mem0.add([{"role": "user", "content": content}], user_id=user_id)
        return {"status": "success", "message": "Information saved to memory"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save memory: {str(e)}"}

personal_assistant = Agent(
    name="personal_assistant",
    model="gemini-2.0-flash",
    instruction="""You are a helpful personal assistant with memory capabilities.
    Use the search_memory function to recall past conversations and user preferences.
    Use the save_memory function to store important information about the user.
    Always personalize your responses based on available memory.""",
    description="A personal assistant that remembers user preferences and past interactions",
    tools=[search_memory, save_memory]
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
    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name="memory_assistant",
        user_id=user_id,
        session_id=f"session_{user_id}"
    )
    runner = Runner(agent=personal_assistant, app_name="memory_assistant", session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=user_input)])
    events = runner.run(user_id=user_id, session_id=session.id, new_message=content)

    for event in events:
        if event.is_final_response():
            response = event.content.parts[0].text

            return response

    return "No response generated"

if __name__ == "__main__":
    response = chat_with_agent(
        "I love Italian food and I'm planning a trip to Rome next month",
        user_id="alice"
    )
    logger.debug(response)

"""
## Multi-Agent Hierarchy with Shared Memory

Create specialized agents in a hierarchy that share memory:
"""
logger.info("## Multi-Agent Hierarchy with Shared Memory")


travel_agent = Agent(
    name="travel_specialist",
    model="gemini-2.0-flash",
    instruction="""You are a travel planning specialist. Use get_user_context to
    understand the user's travel preferences and history before making recommendations.
    After providing advice, use store_interaction to save travel-related information.""",
    description="Specialist in travel planning and recommendations",
    tools=[search_memory, save_memory]
)

health_agent = Agent(
    name="health_advisor",
    model="gemini-2.0-flash",
    instruction="""You are a health and wellness advisor. Use get_user_context to
    understand the user's health goals and dietary preferences.
    After providing advice, use store_interaction to save health-related information.""",
    description="Specialist in health and wellness advice",
    tools=[search_memory, save_memory]
)

coordinator_agent = Agent(
    name="coordinator",
    model="gemini-2.0-flash",
    instruction="""You are a coordinator that delegates requests to specialist agents.
    For travel-related questions (trips, hotels, flights, destinations), delegate to the travel specialist.
    For health-related questions (fitness, diet, wellness, exercise), delegate to the health advisor.
    Use get_user_context to understand the user before delegation.""",
    description="Coordinates requests between specialist agents",
    tools=[
        AgentTool(agent=travel_agent, skip_summarization=False),
        AgentTool(agent=health_agent, skip_summarization=False)
    ]
)

def chat_with_specialists(user_input: str, user_id: str) -> str:
    """
    Handle user input with specialist agent delegation and memory.

    Args:
        user_input: The user's message
        user_id: Unique identifier for the user

    Returns:
        The specialist agent's response
    """
    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name="specialist_system",
        user_id=user_id,
        session_id=f"session_{user_id}"
    )
    runner = Runner(agent=coordinator_agent, app_name="specialist_system", session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=user_input)])
    events = runner.run(user_id=user_id, session_id=session.id, new_message=content)

    for event in events:
        if event.is_final_response():
            response = event.content.parts[0].text

            conversation = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ]
            mem0.add(conversation, user_id=user_id)

            return response

    return "No response generated"

response = chat_with_specialists("Plan a healthy meal for my Italy trip", user_id="alice")
logger.debug(response)

"""
## Quick Start Chat Interface

Simple interactive chat with memory and Google ADK:
"""
logger.info("## Quick Start Chat Interface")

def interactive_chat():
    """Interactive chat interface with memory and ADK"""
    user_id = input("Enter your user ID: ") or "demo_user"
    logger.debug(f"Chat started for user: {user_id}")
    logger.debug("Type 'quit' to exit")
    logger.debug("=" * 50)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == 'quit':
            logger.debug("Goodbye! Your conversation has been saved to memory.")
            break
        else:
            response = chat_with_specialists(user_input, user_id)
            logger.debug(f"Assistant: {response}")

if __name__ == "__main__":
    interactive_chat()

"""
## Key Features

### 1. Memory-Enhanced Function Tools
- **Function Tools**: Standard Python functions that can search and save memories
- **Tool Context**: Access to session state and memory through function parameters
- **Structured Returns**: Dictionary-based returns with status indicators for better LLM understanding

### 2. Multi-Agent Memory Sharing
- **Agent-as-a-Tool**: Specialists can be called as tools while maintaining shared memory
- **Hierarchical Delegation**: Coordinator agents route to specialists based on context
- **Memory Categories**: Store interactions with metadata for better organization

### 3. Flexible Memory Operations
- **Search Capabilities**: Retrieve relevant memories through conversation history
- **User Segmentation**: Organize memories by user ID
- **Memory Management**: Built-in tools for saving and retrieving information

## Configuration Options

Customize memory behavior and agent setup:
"""
logger.info("## Key Features")

memories = mem0.search(
    query="travel preferences",
    user_id="alice",
    limit=5,
    filters={"category": "travel"}  # Filter by category if supported
)

agent = Agent(
    name="custom_agent",
    model="gemini-2.0-flash",  # or use LiteLLM for other models
    instruction="Custom agent behavior",
    tools=[memory_tools],
)

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

"""
## Help

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Mem0 Platform](https://app.mem0.ai/)
- If you need further assistance, please feel free to reach out to us through the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Help")

logger.info("\n\n[DONE]", bright=True)