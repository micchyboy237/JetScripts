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
---
title: Google AI
---

To use Google AI embedding models, set the `GOOGLE_API_KEY` environment variables. You can obtain the Gemini API key from [here](https://aistudio.google.com/app/apikey).

### Usage
"""
logger.info("### Usage")


os.environ["GOOGLE_API_KEY"] = "key"
# os.environ["OPENAI_API_KEY"] = "your_api_key" # For LLM

config = {
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
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

Here are the parameters available for configuring Gemini embedder:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `model` | The name of the embedding model to use | `models/text-embedding-004` |
| `embedding_dims` | Dimensions of the embedding model (output_dimensionality will be considered as embedding_dims, so please set embedding_dims accordingly) | `768` |
| `api_key` | The Google API key | `None` |
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)