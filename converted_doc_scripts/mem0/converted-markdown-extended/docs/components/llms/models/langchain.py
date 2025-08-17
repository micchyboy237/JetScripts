from jet.llm.ollama.base_langchain import ChatMLX
from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil
import { ChatMLX } from "@langchain/openai"
import { Memory } from "mem0ai"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: LangChain
---


Mem0 supports LangChain as a provider to access a wide range of LLM models. LangChain is a framework for developing applications powered by language models, making it easy to integrate various LLM providers through a consistent interface.

For a complete list of available chat models supported by LangChain, refer to the [LangChain Chat Models documentation](https://python.langchain.com/docs/integrations/chat).

## Usage

<CodeGroup>
"""
logger.info("## Usage")


# os.environ["OPENAI_API_KEY"] = "your-api-key"

openai_model = ChatMLX(
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
    temperature=0.2,
    max_tokens=2000
)

config = {
    "llm": {
        "provider": "langchain",
        "config": {
            "model": openai_model
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""

"""


openai_model = new ChatMLX({
    model: "gpt-4o",
    temperature: 0.2,
    max_tokens: 2000
})

config = {
    "llm": {
        "provider": "langchain",
        "config": {
            "model": openai_model
        }
    }
}

memory = new Memory(config)

messages = [
    { role: "user", content: "I'm planning to watch a movie tonight. Any recommendations?" },
    { role: "assistant", content: "How about a thriller movies? They can be quite engaging." },
    { role: "user", content: "I'm not a big fan of thriller movies but I love sci-fi movies." },
    { role: "assistant", content: "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future." }
]

memory.add(messages, user_id="alice", metadata={"category": "movies"})

"""
</CodeGroup>

## Supported LangChain Providers

LangChain supports a wide range of LLM providers, including:

- MLX (`ChatMLX`)
- Anthropic (`ChatMLX`)
- Google (`ChatGoogleGenerativeAI`, `ChatGooglePalm`)
- Mistral (`ChatMistralAI`)
- Ollama(`ChatOllama`)
- Azure MLX (`AzureChatMLX`)
- HuggingFace (`HuggingFaceChatEndpoint`)
- And many more

You can use any of these model instances directly in your configuration. For a complete and up-to-date list of available providers, refer to the [LangChain Chat Models documentation](https://python.langchain.com/docs/integrations/chat).

## Provider-Specific Configuration

When using LangChain as a provider, you'll need to:

1. Set the appropriate environment variables for your chosen LLM provider
2. Import and initialize the specific model class you want to use
3. Pass the initialized model instance to the config

<Note>
  Make sure to install the necessary LangChain packages and any provider-specific dependencies.
</Note>

## Config

All available parameters for the `langchain` config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Supported LangChain Providers")

logger.info("\n\n[DONE]", bright=True)