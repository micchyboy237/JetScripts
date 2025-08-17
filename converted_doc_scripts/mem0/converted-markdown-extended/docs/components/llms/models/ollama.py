from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
You can use LLMs from Ollama to run Mem0 locally. These [models](https://ollama.com/search?c=tools) support tool support.

## Usage
"""
logger.info("## Usage")


# os.environ["OPENAI_API_KEY"] = "your-api-key" # for embedder

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "mixtral:8x7b",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""
## Config

All available parameters for the `ollama` config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Config")

logger.info("\n\n[DONE]", bright=True)