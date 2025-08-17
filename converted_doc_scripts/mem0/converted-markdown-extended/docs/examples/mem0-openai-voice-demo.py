import asyncio
from jet.transformers.formatters import format_json
from agents import (
Agent,
function_tool
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import (
AudioInput,
SingleAgentVoiceWorkflow,
VoicePipeline
)
from jet.logger import CustomLogger
from mem0 import AsyncMemoryClient
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import logging
import numpy as np
import os
import shutil
import sounddevice as sd


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: 'Mem0 with MLX Agents SDK for Voice'
description: 'Integrate memory capabilities into your voice agents using Mem0 and MLX Agents SDK'
---

# Building Voice Agents with Memory using Mem0 and MLX Agents SDK

This guide demonstrates how to combine MLX's Agents SDK for voice applications with Mem0's memory capabilities to create a voice assistant that remembers user preferences and past interactions.

## Prerequisites

Before you begin, make sure you have:

1. Installed MLX Agents SDK with voice dependencies:
"""
logger.info("# Building Voice Agents with Memory using Mem0 and MLX Agents SDK")

pip install 'openai-agents[voice]'

"""
2. Installed Mem0 SDK:
"""
logger.info("2. Installed Mem0 SDK:")

pip install mem0ai

"""
3. Installed other required dependencies:
"""
logger.info("3. Installed other required dependencies:")

pip install numpy sounddevice pydantic

"""
4. Set up your API keys:
   - MLX API key for the Agents SDK
   - Mem0 API key from the Mem0 Platform

## Code Breakdown

Let's break down the key components of this implementation:

### 1. Setting Up Dependencies and Environment
"""
logger.info("## Code Breakdown")



# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["MEM0_API_KEY"] = "your-mem0-api-key"

USER_ID = "voice_user"

mem0_client = AsyncMemoryClient()

"""
This section handles:
- Importing required modules from MLX Agents SDK and Mem0
- Setting up environment variables for API keys
- Defining a simple user identification system (using a global variable)
- Initializing the Mem0 client that will handle memory operations

### 2. Memory Tools with Function Decorators

The `@function_tool` decorator transforms Python functions into callable tools for the MLX agent. Here are the key memory tools:

#### Storing User Memories
"""
logger.info("### 2. Memory Tools with Function Decorators")


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger("memory_voice_agent")

@function_tool
async def save_memories(
    memory: str
) -> str:
    """Store a user memory in memory."""
    logger.debug(f"Saving memory: {memory} for user {USER_ID}")

    memory_content = f"User memory - {memory}"
    await mem0_client.add(
        memory_content,
        user_id=USER_ID,
    )

    return f"I've saved your memory: {memory}"

"""
This function:
- Takes a memory string
- Creates a formatted memory string
- Stores it in Mem0 using the `add()` method
- Includes metadata to categorize the memory for easier retrieval
- Returns a confirmation message that the agent will speak

#### Finding Relevant Memories
"""
logger.info("#### Finding Relevant Memories")

@function_tool
async def search_memories(
    query: str
) -> str:
    """
    Find memories relevant to the current conversation.
    Args:
        query: The search query to find relevant memories
    """
    logger.debug(f"Finding memories related to: {query}")
    async def async_func_10():
        results = await mem0_client.search(
            query,
            user_id=USER_ID,
            limit=5,
            threshold=0.7,  # Higher threshold for more relevant results
            output_format="v1.1"
        )
        return results
    results = asyncio.run(async_func_10())
    logger.success(format_json(results))

    if not results.get('results', []):
        return "I don't have any relevant memories about this topic."

    memories = [f"• {result['memory']}" for result in results.get('results', [])]
    return "Here's what I remember that might be relevant:\n" + "\n".join(memories)

"""
This tool:
- Takes a search query string
- Passes it to Mem0's semantic search to find related memories
- Sets a threshold for relevance to ensure quality results
- Returns a formatted list of relevant memories or a default message

### 3. Creating the Voice Agent
"""
logger.info("### 3. Creating the Voice Agent")

def create_memory_voice_agent():
    agent = Agent(
        name="Memory Assistant",
        instructions=prompt_with_handoff_instructions(
            """You're speaking to a human, so be polite and concise.
            Always respond in clear, natural English.
            You have the ability to remember information about the user.
            Use the save_memories tool when the user shares an important information worth remembering.
            Use the search_memories tool when you need context from past conversations or user asks you to recall something.
            """,
        ),
        model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
        tools=[save_memories, search_memories],
    )

    return agent

"""
This function:
- Creates an MLX Agent with specific instructions
- Configures it to use gpt-4o (you can use other models)
- Registers the memory-related tools with the agent
- Uses `prompt_with_handoff_instructions` to include standard voice agent behaviors

### 4. Microphone Recording Functionality
"""
logger.info("### 4. Microphone Recording Functionality")

async def record_from_microphone(duration=5, samplerate=24000):
    """Record audio from the microphone for a specified duration."""
    logger.debug(f"Recording for {duration} seconds...")

    frames = []

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, dtype=np.int16):
        async def run_async_code_f93d9ff2():
            await asyncio.sleep(duration)
            return 
         = asyncio.run(run_async_code_f93d9ff2())
        logger.success(format_json())

    audio_data = np.concatenate(frames)
    return audio_data

"""
This function:
- Creates a simple asynchronous microphone recording function
- Uses the sounddevice library to capture audio input
- Stores frames in a buffer during recording
- Combines frames into a single numpy array when complete
- Returns the audio data for processing

### 5. Main Loop and Voice Processing
"""
logger.info("### 5. Main Loop and Voice Processing")

async def main():
    agent = create_memory_voice_agent()

    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent)
    )

    pipeline.config.tts_settings.voice = "alloy"
    pipeline.config.tts_settings.speed = 1.0

    try:
        while True:
            logger.debug("\nPress Enter to start recording (or 'q' to quit)...")
            user_input = input()
            if user_input.lower() == 'q':
                break

            async def run_async_code_a214b85e():
                async def run_async_code_1c7acb0b():
                    audio_data = await record_from_microphone(duration=5)
                    return audio_data
                audio_data = asyncio.run(run_async_code_1c7acb0b())
                logger.success(format_json(audio_data))
                return audio_data
            audio_data = asyncio.run(run_async_code_a214b85e())
            logger.success(format_json(audio_data))
            audio_input = AudioInput(buffer=audio_data)
            async def run_async_code_4b51a16c():
                async def run_async_code_019a24e9():
                    result = await pipeline.run(audio_input)
                    return result
                result = asyncio.run(run_async_code_019a24e9())
                logger.success(format_json(result))
                return result
            result = asyncio.run(run_async_code_4b51a16c())
            logger.success(format_json(result))

            player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
            player.start()

            agent_response = ""
            logger.debug("\nAgent response:")

            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    player.write(event.data)
                elif event.type == "voice_stream_event_content":
                    content = event.data
                    agent_response += content
                    logger.debug(content, end="", flush=True)

            if agent_response:
                try:
                    await mem0_client.add(
                        f"Agent response: {agent_response}",
                        user_id=USER_ID,
                        metadata={"type": "agent_response"}
                    )
                except Exception as e:
                    logger.debug(f"Failed to store memory: {e}")

    except KeyboardInterrupt:
        logger.debug("\nExiting...")

"""
This main function orchestrates the entire process:
1. Creates the memory-enabled voice agent
2. Sets up the voice pipeline with TTS settings
3. Implements an interactive loop for recording and processing voice input
4. Handles streaming of response events (both audio and text)
5. Automatically saves the agent's responses to memory
6. Includes proper error handling and exit mechanisms

## Create a Memory-Enabled Voice Agent

Now that we've explained each component, here's the complete implementation that combines MLX Agents SDK for voice with Mem0's memory capabilities:
"""
logger.info("## Create a Memory-Enabled Voice Agent")




# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["MEM0_API_KEY"] = "your-mem0-api-key"

USER_ID = "voice_user"

mem0_client = AsyncMemoryClient()

@function_tool
async def save_memories(
    memory: str
) -> str:
    """
    Store a user memory in memory.
    Args:
        memory: The memory to save
    """
    logger.debug(f"Saving memory: {memory} for user {USER_ID}")

    memory_content = f"User memory - {memory}"
    await mem0_client.add(
        memory_content,
        user_id=USER_ID,
    )

    return f"I've saved your memory: {memory}"

@function_tool
async def search_memories(
    query: str
) -> str:
    """
    Find memories relevant to the current conversation.
    Args:
        query: The search query to find relevant memories
    """
    logger.debug(f"Finding memories related to: {query}")
    async def async_func_57():
        results = await mem0_client.search(
            query,
            user_id=USER_ID,
            limit=5,
            threshold=0.7,  # Higher threshold for more relevant results
            output_format="v1.1"
        )
        return results
    results = asyncio.run(async_func_57())
    logger.success(format_json(results))

    if not results.get('results', []):
        return "I don't have any relevant memories about this topic."

    memories = [f"• {result['memory']}" for result in results.get('results', [])]
    return "Here's what I remember that might be relevant:\n" + "\n".join(memories)

def create_memory_voice_agent():
    agent = Agent(
        name="Memory Assistant",
        instructions=prompt_with_handoff_instructions(
            """You're speaking to a human, so be polite and concise.
            Always respond in clear, natural English.
            You have the ability to remember information about the user.
            Use the save_memories tool when the user shares an important information worth remembering.
            Use the search_memories tool when you need context from past conversations or user asks you to recall something.
            """,
        ),
        model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
        tools=[save_memories, search_memories],
    )

    return agent

async def record_from_microphone(duration=5, samplerate=24000):
    """Record audio from the microphone for a specified duration."""
    logger.debug(f"Recording for {duration} seconds...")

    frames = []

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, dtype=np.int16):
        async def run_async_code_f93d9ff2():
            await asyncio.sleep(duration)
            return 
         = asyncio.run(run_async_code_f93d9ff2())
        logger.success(format_json())

    audio_data = np.concatenate(frames)
    return audio_data

async def main():
    logger.debug("Starting Memory Voice Agent")

    agent = create_memory_voice_agent()

    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent)
    )

    pipeline.config.tts_settings.voice = "alloy"
    pipeline.config.tts_settings.speed = 1.0

    try:
        while True:
            logger.debug("\nPress Enter to start recording (or 'q' to quit)...")
            user_input = input()
            if user_input.lower() == 'q':
                break

            async def run_async_code_a214b85e():
                async def run_async_code_1c7acb0b():
                    audio_data = await record_from_microphone(duration=5)
                    return audio_data
                audio_data = asyncio.run(run_async_code_1c7acb0b())
                logger.success(format_json(audio_data))
                return audio_data
            audio_data = asyncio.run(run_async_code_a214b85e())
            logger.success(format_json(audio_data))
            audio_input = AudioInput(buffer=audio_data)

            logger.debug("Processing your request...")

            async def run_async_code_4b51a16c():
                async def run_async_code_019a24e9():
                    result = await pipeline.run(audio_input)
                    return result
                result = asyncio.run(run_async_code_019a24e9())
                logger.success(format_json(result))
                return result
            result = asyncio.run(run_async_code_4b51a16c())
            logger.success(format_json(result))

            player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
            player.start()

            agent_response = ""

            logger.debug("\nAgent response:")
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    player.write(event.data)
                elif event.type == "voice_stream_event_content":
                    content = event.data
                    agent_response += content
                    logger.debug(content, end="", flush=True)

            logger.debug("\n")

            if agent_response:
                try:
                    await mem0_client.add(
                        f"Agent response: {agent_response}",
                        user_id=USER_ID,
                        metadata={"type": "agent_response"}
                    )
                except Exception as e:
                    logger.debug(f"Failed to store memory: {e}")

    except KeyboardInterrupt:
        logger.debug("\nExiting...")

if __name__ == "__main__":
    asyncio.run(main())

"""
## Key Features of This Implementation

This implementation offers several key features:

1. **Simplified User Management**: Uses a global `USER_ID` variable for simplicity, but can be extended to manage multiple users.

2. **Real Microphone Input**: Includes a `record_from_microphone()` function that captures actual voice input from your microphone.

3. **Interactive Voice Loop**: Implements a continuous interaction loop, allowing for multiple back-and-forth exchanges.

4. **Memory Management Tools**:
   - `save_memories`: Stores user memories in Mem0
   - `search_memories`: Searches for relevant past information

5. **Voice Configuration**: Demonstrates how to configure TTS settings for the voice response.

## Running the Example

To run this example:

1. Replace the placeholder API keys with your actual keys
2. Make sure your microphone is properly connected
3. Run the script with Python 3.8 or newer
4. Press Enter to start recording, then speak your request
5. Press 'q' to quit the application

The agent will listen to your request, process it through the MLX model, utilize Mem0 for memory operations as needed, and respond both through text output and voice speech.

## Best Practices for Voice Agents with Memory

1. **Optimizing Memory for Voice**: Keep memories concise and relevant for voice responses.

2. **Forgetting Mechanism**: Implement a way to delete or expire memories that are no longer relevant.

3. **Context Preservation**: Store enough context with each memory to make retrieval effective.

4. **Error Handling**: Implement robust error handling for memory operations, as voice interactions should continue smoothly even if memory operations fail.

## Conclusion

By combining MLX's Agents SDK with Mem0's memory capabilities, you can create voice agents that maintain persistent memory of user preferences and past interactions. This significantly enhances the user experience by making conversations more natural and personalized.

As you build your voice application, experiment with different memory strategies and filtering approaches to find the optimal balance between comprehensive memory and efficient retrieval for your specific use case.

## Debugging Function Tools

When working with the MLX Agents SDK, you might notice that regular `logger.debug()` statements inside `@function_tool` decorated functions don't appear in your console output. This is because the Agents SDK captures and redirects standard output when executing these functions.

To effectively debug your function tools, use Python's `logging` module instead:
"""
logger.info("## Key Features of This Implementation")


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger("memory_voice_agent")

@function_tool
async def save_memories(
    memory: str
) -> str:
    """Store a user memory in memory."""
    logger.debug(f"Saving memory: {memory} for user {USER_ID}")

logger.info("\n\n[DONE]", bright=True)