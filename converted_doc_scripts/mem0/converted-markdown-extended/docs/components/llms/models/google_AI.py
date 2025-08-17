import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil
import { Memory } from "mem0ai/oss"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Google AI
---

To use the Gemini model, set the `GOOGLE_API_KEY` environment variable. You can obtain the Google/Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

> **Note:** As of the latest release, Mem0 uses the new `google.genai` SDK instead of the deprecated `google.generativeai`. All message formatting and model interaction now use the updated `types` module from `google.genai`.

> **Note:** Some Gemini models are being deprecated and will retire soon. It is recommended to migrate to the latest stable models like `"gemini-2.0-flash-001"` or `"gemini-2.0-flash-lite-001"` to ensure ongoing support and improvements.

## Usage

<CodeGroup>
"""
logger.info("## Usage")


# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Used for embedding model
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"

config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash-001",
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0
        }
    }
}

m = Memory.from_config(config)

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thrillers, but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thrillers and suggest sci-fi movies instead."}
]

m.add(messages, user_id="alice", metadata={"category": "movies"})


config = {
    llm: {
        provider: "gemini",
        config: {
            model: "gemini-2.0-flash-001",
            temperature: 0.1
        }
    }
}

memory = new Memory(config)

messages = [
    { role: "user", content: "I'm planning to watch a movie tonight. Any recommendations?" },
    { role: "assistant", content: "How about thriller movies? They can be quite engaging." },
    { role: "user", content: "I’m not a big fan of thrillers, but I love sci-fi movies." },
    { role: "assistant", content: "Got it! I'll avoid thrillers and suggest sci-fi movies instead." }
]

async def run_async_code_3ea57109():
    await memory.add(messages, { userId: "alice", metadata: { category: "movies" } })
    return 
 = asyncio.run(run_async_code_3ea57109())
logger.success(format_json())

"""
</CodeGroup>

## Config

All available parameters for the `Gemini` config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Config")

logger.info("\n\n[DONE]", bright=True)