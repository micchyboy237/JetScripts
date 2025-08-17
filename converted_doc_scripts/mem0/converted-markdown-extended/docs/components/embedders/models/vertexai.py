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
### Vertex AI

To use Google Cloud's Vertex AI for text embedding models, set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to the path of your service account's credentials JSON file. These credentials can be created in the [Google Cloud Console](https://console.cloud.google.com/).

### Usage
"""
logger.info("### Vertex AI")


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/credentials.json"
# os.environ["OPENAI_API_KEY"] = "your_api_key" # For LLM

config = {
    "embedder": {
        "provider": "vertexai",
        "config": {
            "model": "text-embedding-004",
            "memory_add_embedding_type": "RETRIEVAL_DOCUMENT",
            "memory_update_embedding_type": "RETRIEVAL_DOCUMENT",
            "memory_search_embedding_type": "RETRIEVAL_QUERY"
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
The embedding types can be one of the following:
- SEMANTIC_SIMILARITY
- CLASSIFICATION
- CLUSTERING
- RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION
- CODE_RETRIEVAL_QUERY  
Check out the [Vertex AI documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types#supported_task_types) for more information.  
  
### Config

Here are the parameters available for configuring the Vertex AI embedder:

| Parameter                 | Description                                      | Default Value        |
| ------------------------- | ------------------------------------------------ | -------------------- |
| `model`                   | The name of the Vertex AI embedding model to use | `text-embedding-004` |
| `vertex_credentials_json` | Path to the Google Cloud credentials JSON file   | `None`               |
| `embedding_dims`          | Dimensions of the embedding model                | `256`                |
| `memory_add_embedding_type` | The type of embedding to use for the add memory action | `RETRIEVAL_DOCUMENT` |
| `memory_update_embedding_type` | The type of embedding to use for the update memory action | `RETRIEVAL_DOCUMENT` |
| `memory_search_embedding_type` | The type of embedding to use for the search memory action | `RETRIEVAL_QUERY` |
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)