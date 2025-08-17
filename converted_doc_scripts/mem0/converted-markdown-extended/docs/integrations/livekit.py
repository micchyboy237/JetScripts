import asyncio
from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from jet.logger import CustomLogger
from livekit.agents import (
JobContext,
WorkerOptions,
cli,
ChatContext,
ChatMessage,
RoomInputOptions,
Agent,
AgentSession,
)
from livekit.plugins import openai, silero, deepgram, noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel
from mem0 import AsyncMemoryClient
from pathlib import Path
import logging
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
title: Livekit
---

This guide demonstrates how to create a memory-enabled voice assistant using LiveKit, Deepgram, MLX, and Mem0, focusing on creating an intelligent, context-aware travel planning agent.

## Prerequisites

Before you begin, make sure you have:

1. Installed Livekit Agents SDK with voice dependencies of silero and deepgram:
"""
logger.info("## Prerequisites")

pip install livekit livekit-agents \
livekit-plugins-silero \
livekit-plugins-deepgram \
livekit-plugins-openai \
livekit-plugins-turn-detector \
livekit-plugins-noise-cancellation

"""
2. Installed Mem0 SDK:
"""
logger.info("2. Installed Mem0 SDK:")

pip install mem0ai

"""
3. Set up your API keys in a `.env` file:
"""
logger.info("3. Set up your API keys in a `.env` file:")

LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
DEEPGRAM_API_KEY=your_deepgram_api_key
MEM0_API_KEY=your_mem0_api_key
# OPENAI_API_KEY=your_openai_api_key

"""
> **Note**: Make sure to have a Livekit and Deepgram account. You can find these variables `LIVEKIT_URL` , `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` from [LiveKit Cloud Console](https://cloud.livekit.io/) and for more information you can refer this website [LiveKit Documentation](https://docs.livekit.io/home/cloud/keys-and-tokens/). For `DEEPGRAM_API_KEY` you can get from [Deepgram Console](https://console.deepgram.com/) refer this website [Deepgram Documentation](https://developers.deepgram.com/docs/create-additional-api-keys) for more details.

## Code Breakdown

Let's break down the key components of this implementation using LiveKit Agents:

### 1. Setting Up Dependencies and Environment
"""
logger.info("## Code Breakdown")




load_dotenv()

"""
### 2. Mem0 Client and Agent Definition
"""
logger.info("### 2. Mem0 Client and Agent Definition")

RAG_USER_ID = "livekit-mem0"
mem0_client = AsyncMemoryClient()

class MemoryEnabledAgent(Agent):
    """
    An agent that can answer questions using RAG (Retrieval Augmented Generation) with Mem0.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful voice assistant.
                You are a travel guide named George and will help the user to plan a travel trip of their dreams.
                You should help the user plan for various adventures like work retreats, family vacations or solo backpacking trips.
                You should be careful to not suggest anything that would be dangerous, illegal or inappropriate.
                You can remember past interactions and use them to inform your answers.
                Use semantic memory retrieval to provide contextually relevant responses.
            """,
        )
        self._seen_results = set()  # Track previously seen result IDs
        logger.info(f"Mem0 Agent initialized. Using user_id: {RAG_USER_ID}")

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Briefly greet the user and offer your assistance."
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        try:
            logger.info(f"Adding user message to Mem0: {new_message.text_content}")
            async def async_func_29():
                add_result = await mem0_client.add(
                    [{"role": "user", "content": new_message.text_content}],
                    user_id=RAG_USER_ID
                )
                return add_result
            add_result = asyncio.run(async_func_29())
            logger.success(format_json(add_result))
            logger.info(f"Mem0 add result (user): {add_result}")
        except Exception as e:
            logger.warning(f"Failed to store user message in Mem0: {e}")

        try:
            async def run_async_code_b12bb432():
                logger.info("About to await mem0_client.search for RAG context")
                return 
             = asyncio.run(run_async_code_b12bb432())
            logger.success(format_json())
            async def async_func_39():
                search_results = await mem0_client.search(
                    new_message.text_content,
                    user_id=RAG_USER_ID,
                )
                return search_results
            search_results = asyncio.run(async_func_39())
            logger.success(format_json(search_results))
            logger.info(f"mem0_client.search returned: {search_results}")
            if search_results and search_results.get('results', []):
                context_parts = []
                for result in search_results.get('results', []):
                    paragraph = result.get("memory") or result.get("text")
                    if paragraph:
                        source = "mem0 Memories"
                        if "from [" in paragraph:
                            source = paragraph.split("from [")[1].split("]")[0]
                            paragraph = paragraph.split("]")[1].strip()
                        context_parts.append(f"Source: {source}\nContent: {paragraph}\n")
                if context_parts:
                    full_context = "\n\n".join(context_parts)
                    logger.info(f"Injecting RAG context: {full_context}")
                    turn_ctx.add_message(role="assistant", content=full_context)
                    async def run_async_code_e06f7e65():
                        await self.update_chat_ctx(turn_ctx)
                        return 
                     = asyncio.run(run_async_code_e06f7e65())
                    logger.success(format_json())
        except Exception as e:
            logger.warning(f"Failed to inject RAG context from Mem0: {e}")

        async def run_async_code_c716529d():
            await super().on_user_turn_completed(turn_ctx, new_message)
            return 
         = asyncio.run(run_async_code_c716529d())
        logger.success(format_json())

"""
### 3. Entrypoint and Session Setup
"""
logger.info("### 3. Entrypoint and Session Setup")

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""
    async def run_async_code_1b94d4e9():
        await ctx.connect()
        return 
     = asyncio.run(run_async_code_1b94d4e9())
    logger.success(format_json())

    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="llama-3.2-3b-instruct"),
        tts=openai.TTS(voice="ash",),
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=MemoryEnabledAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user warmly as George the travel guide and ask how you can help them plan their next adventure.",
        allow_interruptions=True
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

"""
## Key Features of This Implementation

1. **Semantic Memory Retrieval**: Uses Mem0 to store and retrieve contextually relevant memories
2. **Voice Interaction**: Leverages LiveKit for voice communication with proper turn detection
3. **Intelligent Context Management**: Augments conversations with past interactions
4. **Travel Planning Specialization**: Focused on creating a helpful travel guide assistant
5. **Function Tools**: Modern tool definition for enhanced capabilities

## Running the Example

To run this example:

1. Install all required dependencies
2. Set up your `.env` file with the necessary API keys
3. Ensure your microphone and audio setup are configured
4. Run the script with Python 3.11 or newer and with the following command:
"""
logger.info("## Key Features of This Implementation")

python mem0-livekit-voice-agent.py start

"""
or to start your agent in console mode to run inside your terminal:
"""
logger.info("or to start your agent in console mode to run inside your terminal:")

python mem0-livekit-voice-agent.py console

"""
5. After the script starts, you can interact with the voice agent using [Livekit's Agent Platform](https://agents-playground.livekit.io/) and connect to the agent inorder to start conversations.

## Best Practices for Voice Agents with Memory

1. **Context Preservation**: Store enough context with each memory for effective retrieval
2. **Privacy Considerations**: Implement secure memory management
3. **Relevant Memory Filtering**: Use semantic search to retrieve only the most relevant memories
4. **Error Handling**: Implement robust error handling for memory operations

## Debugging Function Tools

- To run the script in debug mode simply start the assistant with `dev` mode:
"""
logger.info("## Best Practices for Voice Agents with Memory")

python mem0-livekit-voice-agent.py dev

"""
- When working with memory-enabled voice agents, use Python's `logging` module for effective debugging:
"""


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_voice_agent")

"""
- Check the logs for any issues with API keys, connectivity, or memory operations.
- Ensure your `.env` file is correctly configured and loaded.


## Help & Resources

- [LiveKit Documentation](https://docs.livekit.io/)
- [Mem0 Platform](https://app.mem0.ai/)
- Need assistance? Reach out through:

<Snippet file="get-help.mdx" />
"""
logger.info("## Help & Resources")

logger.info("\n\n[DONE]", bright=True)