from jet.logger import CustomLogger
from mem0.proxy.main import Mem0
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
title: MLX Compatibility
icon: "code"
iconType: "solid"
---

Mem0 can be easily integrated into chat applications to enhance conversational agents with structured memory. Mem0's APIs are designed to be compatible with MLX's, with the goal of making it easy to leverage Mem0 in applications you may have already built.

If you have a `Mem0 API key`, you can use it to initialize the client. Alternatively, you can initialize Mem0 without an API key if you're using it locally.

Mem0 supports several language models (LLMs) through integration with various [providers](https://litellm.vercel.app/docs/providers).

## Use Mem0 Platform
"""
logger.info("## Use Mem0 Platform")


client = Mem0(api_key="m0-xxx")

messages = [
    {
        "role": "user",
        "content": "I love indian food but I cannot eat pizza since allergic to cheese."
    },
]
user_id = "alice"
chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.2-3b-instruct",
    user_id=user_id
)

messages = [
    {
        "role": "user",
        "content": "Suggest restaurants in San Francisco to eat.",
    }
]

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.2-3b-instruct",
    user_id=user_id
)
logger.debug(chat_completion.choices[0].message.content)

"""
In this example, you can see how the second response is tailored based on the information provided in the first interaction. Mem0 remembers the user's preference for Indian food and their cheese allergy, using this information to provide more relevant and personalized restaurant suggestions in San Francisco.

### Use Mem0 OSS
"""
logger.info("### Use Mem0 OSS")

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    },
}

client = Mem0(config=config)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What's the capital of France?",
        }
    ],
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
)

"""
## Mem0 Params for Chat Completion

- `user_id` (Optional[str]): Identifier for the user.

- `agent_id` (Optional[str]): Identifier for the agent.

- `run_id` (Optional[str]): Identifier for the run.

- `metadata` (Optional[dict]): Additional metadata to be stored with the memory.

- `filters` (Optional[dict]): Filters to apply when searching for relevant memories.

- `limit` (Optional[int]): Maximum number of relevant memories to retrieve. Default is 10.


Other parameters are similar to MLX's API, making it easy to integrate Mem0 into your existing applications.
"""
logger.info("## Mem0 Params for Chat Completion")

logger.info("\n\n[DONE]", bright=True)