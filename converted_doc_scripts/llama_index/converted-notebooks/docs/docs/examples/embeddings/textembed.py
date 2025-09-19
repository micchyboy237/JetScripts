from jet.logger import CustomLogger
from llama_index.embeddings.textembed import TextEmbedEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# TextEmbed - Embedding Inference Server

Maintained by Keval Dekivadiya, TextEmbed is licensed under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0).

TextEmbed is a high-throughput, low-latency REST API designed for serving vector embeddings. It supports a wide range of sentence-transformer models and frameworks, making it suitable for various applications in natural language processing.

## Features

- **High Throughput & Low Latency**: Designed to handle a large number of requests efficiently.
- **Flexible Model Support**: Works with various sentence-transformer models.
- **Scalable**: Easily integrates into larger systems and scales with demand.
- **Batch Processing**: Supports batch processing for better and faster inference.
- **OllamaFunctionCalling Compatible REST API Endpoint**: Provides an OllamaFunctionCalling compatible REST API endpoint.
- **Single Line Command Deployment**: Deploy multiple models via a single command for efficient deployment.
- **Support for Embedding Formats**: Supports binary, float16, and float32 embeddings formats for faster retrieval.

## Getting Started

### Prerequisites

Ensure you have Python 3.10 or higher installed. You will also need to install the required dependencies.

### Installation via PyPI

Install the required dependencies:
"""
logger.info("# TextEmbed - Embedding Inference Server")

# !pip install -U textembed

"""
### Start the TextEmbed Server

Start the TextEmbed server with your desired models:
"""
logger.info("### Start the TextEmbed Server")

# !python -m textembed.server --models sentence-transformers/all-MiniLM-L12-v2 --workers 4 --api-key TextEmbed

"""
### Example Usage with llama-index

Here's a simple example to get you started with llama-index:
"""
logger.info("### Example Usage with llama-index")


embed = TextEmbedEmbedding(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    base_url="http://0.0.0.0:8000/v1",
    auth_token="TextEmbed",
)

embeddings = embed.get_text_embedding_batch(
    [
        "It is raining cats and dogs here!",
        "India has a diverse cultural heritage.",
    ]
)

logger.debug(embeddings)

"""
For more information, please read the [documentation](https://github.com/kevaldekivadiya2415/textembed/blob/main/docs/setup.md).
"""
logger.info("For more information, please read the [documentation](https://github.com/kevaldekivadiya2415/textembed/blob/main/docs/setup.md).")

logger.info("\n\n[DONE]", bright=True)