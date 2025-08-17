from jet.logger import CustomLogger
from mem0 import Memory
from openai import MLX
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Kuzu as Graph Memory

## Prerequisites

### Install Mem0 with Graph Memory support

To use Mem0 with Graph Memory support, install it using pip:

```bash
pip install "mem0ai[graph]"
```

This command installs Mem0 along with the necessary dependencies for graph functionality.

### Kuzu setup

Kuzu comes embedded into the Python package that gets installed with the above command. There is no extra setup required.
Just pick an empty directory where Kuzu should persist its database.

## Configuration

Do all the imports and configure MLX (enter your MLX API key):
"""
logger.info("# Kuzu as Graph Memory")



# os.environ["OPENAI_API_KEY"] = ""
openai_client = MLX()

"""
Set up configuration to use the embedder model and Neo4j as a graph store:
"""
logger.info("Set up configuration to use the embedder model and Neo4j as a graph store:")

config = {
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-large", "embedding_dims": 1536},
    },
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db": ":memory:",
        },
    },
}
memory = Memory.from_config(config_dict=config)

def print_added_memories(results):
    logger.debug("::: Saved the following memories:")
    logger.debug(" embeddings:")
    for r in results['results']:
        logger.debug("    ",r)
    logger.debug(" relations:")
    for k,v in results['relations'].items():
        logger.debug("    ",k)
        for e in v:
            logger.debug("      ",e)

"""
## Store memories

Create memories:
"""
logger.info("## Store memories")

user = "myuser"

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

"""
Store memories in Kuzu:
"""
logger.info("Store memories in Kuzu:")

results = memory.add(messages, user_id=user, metadata={"category": "movie_recommendations"})
print_added_memories(results)

"""
## Search memories
"""
logger.info("## Search memories")

for result in memory.search("what does alice love?", user_id=user)["results"]:
    logger.debug(result["memory"], result["score"])

"""
## Chatbot
"""
logger.info("## Chatbot")

def chat_with_memories(message: str, user_id: str = user) -> str:
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
    logger.debug("::: Using memories:")
    logger.debug(memories_str)

    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    response = openai_client.chat.completions.create(model="llama-3.2-3b-instruct", messages=messages)
    assistant_response = response.choices[0].message.content

    messages.append({"role": "assistant", "content": assistant_response})
    results = memory.add(messages, user_id=user_id)
    print_added_memories(results)

    return assistant_response

logger.debug("Chat with AI (type 'exit' to quit)")
while True:
    user_input = input(">>> You: ").strip()
    if user_input.lower() == 'exit':
        logger.debug("Goodbye!")
        break
    logger.debug(f"<<< AI response:\n{chat_with_memories(user_input)}")

logger.info("\n\n[DONE]", bright=True)