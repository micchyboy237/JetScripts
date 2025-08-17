import asyncio
from jet.transformers.formatters import format_json
from fastapi import FastAPI, WebSocket
from jet.logger import CustomLogger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.mem0 import Mem0MemoryService
from pipecat.services.openai import MLXLLMService, MLXUserContextAggregator, MLXAssistantContextAggregator
from pipecat.services.whisper import WhisperSTTService
from pipecat.transports.network.fastapi_websocket import (
FastAPIWebsocketTransport,
FastAPIWebsocketParams
)
import asyncio
import os
import shutil
import uvicorn


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: 'Pipecat'
description: 'Integrate Mem0 with Pipecat for conversational memory in AI agents'
---

# Pipecat Integration

Mem0 seamlessly integrates with [Pipecat](https://pipecat.ai), providing long-term memory capabilities for conversational AI agents. This integration allows your Pipecat-powered applications to remember past conversations and provide personalized responses based on user history.

## Installation

To use Mem0 with Pipecat, install the required dependencies:
"""
logger.info("# Pipecat Integration")

pip install "pipecat-ai[mem0]"

"""
You'll also need to set up your Mem0 API key as an environment variable:
"""
logger.info("You'll also need to set up your Mem0 API key as an environment variable:")

export MEM0_API_KEY=your_mem0_api_key

"""
You can obtain a Mem0 API key by signing up at [mem0.ai](https://mem0.ai).

## Configuration

Mem0 integration is provided through the `Mem0MemoryService` class in Pipecat. Here's how to configure it:
"""
logger.info("## Configuration")


memory = Mem0MemoryService(
    api_key=os.getenv("MEM0_API_KEY"),  # Your Mem0 API key
    user_id="unique_user_id",           # Unique identifier for the end user
    agent_id="my_agent",                # Identifier for the agent using the memory
    run_id="session_123",               # Optional: specific conversation session ID
    params={                            # Optional: configuration parameters
        "search_limit": 10,             # Maximum memories to retrieve per query
        "search_threshold": 0.1,        # Relevance threshold (0.0 to 1.0)
        "system_prompt": "Here are your past memories:", # Custom prefix for memories
        "add_as_system_message": True,  # Add memories as system (True) or user (False) message
        "position": 1,                  # Position in context to insert memories
    }
)

"""
## Pipeline Integration

The `Mem0MemoryService` should be positioned between your context aggregator and LLM service in the Pipecat pipeline:
"""
logger.info("## Pipeline Integration")

pipeline = Pipeline([
    transport.input(),
    stt,                # Speech-to-text for audio input
    user_context,       # User context aggregator
    memory,             # Mem0 Memory service enhances context here
    llm,                # LLM for response generation
    tts,                # Optional: Text-to-speech
    transport.output(),
    assistant_context   # Assistant context aggregator
])

"""
## Example: Voice Agent with Memory

Here's a complete example of a Pipecat voice agent with Mem0 memory integration:
"""
logger.info("## Example: Voice Agent with Memory")



app = FastAPI()

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    async def run_async_code_de2c3e20():
        await websocket.accept()
        return 
     = asyncio.run(run_async_code_de2c3e20())
    logger.success(format_json())

    user_id = "user123"

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=ProtobufFrameSerializer(),
        )
    )

    user_context = MLXUserContextAggregator()
    assistant_context = MLXAssistantContextAggregator()
#     stt = WhisperSTTService(api_key=os.getenv("OPENAI_API_KEY"))

    memory = Mem0MemoryService(
        api_key=os.getenv("MEM0_API_KEY"),
        user_id=user_id,
        agent_id="fastapi_memory_bot"
    )

    llm = MLXLLMService(
#         api_key=os.getenv("OPENAI_API_KEY"),
        model="llama-3.2-1b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
        system_prompt="You are a helpful assistant that remembers past conversations."
    )

    pipeline = Pipeline([
        transport.input(),
        stt,                # Speech-to-text for audio input
        user_context,
        memory,             # Memory service enhances context here
        llm,
        transport.output(),
        assistant_context
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        async def run_async_code_1166164b():
            await task.queue_frame(TextFrame("Hello! I'm a memory bot. I'll remember our conversation."))
            return 
         = asyncio.run(run_async_code_1166164b())
        logger.success(format_json())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        async def run_async_code_5a0070d9():
            await task.cancel()
            return 
         = asyncio.run(run_async_code_5a0070d9())
        logger.success(format_json())

    async def run_async_code_49369c92():
        await runner.run(task)
        return 
     = asyncio.run(run_async_code_49369c92())
    logger.success(format_json())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
## How It Works

When integrated with Pipecat, Mem0 provides two key functionalities:

### 1. Message Storage

All conversation messages are automatically stored in Mem0 for future reference:
- Captures the full message history from context frames
- Associates messages with the specified user, agent, and run IDs
- Stores metadata to enable efficient retrieval

### 2. Memory Retrieval

When a new user message is detected:
1. The message is used as a search query to find relevant past memories
2. Relevant memories are retrieved from Mem0's database
3. Memories are formatted and added to the conversation context
4. The enhanced context is passed to the LLM for response generation

## Additional Configuration Options

### Memory Search Parameters

You can customize how memories are retrieved and used:
"""
logger.info("## How It Works")

memory = Mem0MemoryService(
    api_key=os.getenv("MEM0_API_KEY"),
    user_id="user123",
    params={
        "search_limit": 5,            # Retrieve up to 5 memories
        "search_threshold": 0.2,      # Higher threshold for more relevant matches
        "api_version": "v2",          # Mem0 API version
    }
)

"""
### Memory Presentation Options

Control how memories are presented to the LLM:
"""
logger.info("### Memory Presentation Options")

memory = Mem0MemoryService(
    api_key=os.getenv("MEM0_API_KEY"),
    user_id="user123",
    params={
        "system_prompt": "Previous conversations with this user:",
        "add_as_system_message": True,  # Add as system message instead of user message
        "position": 0,                  # Insert at the beginning of the context
    }
)

"""
## Resources

- [Mem0 Pipecat Integration](https://docs.pipecat.ai/server/services/memory/mem0)
- [Pipecat Documentation](https://docs.pipecat.ai)
"""
logger.info("## Resources")

logger.info("\n\n[DONE]", bright=True)