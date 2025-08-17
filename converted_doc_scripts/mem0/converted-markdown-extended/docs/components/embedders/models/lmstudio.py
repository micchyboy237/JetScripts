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
You can use embedding models from LM Studio to run Mem0 locally.

### Usage
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "your_api_key" # For LLM

config = {
    "embedder": {
        "provider": "lmstudio",
        "config": {
            "model": "nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf"
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
m.add(messages, user_id="john")

"""
### Config

Here are the parameters available for configuring Ollama embedder:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `model` | The name of the MLX model to use | `nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf` |
| `embedding_dims` | Dimensions of the embedding model | `1536` |
| `lmstudio_base_url` | Base URL for LM Studio connection | `http://localhost:1234/v1` |
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)