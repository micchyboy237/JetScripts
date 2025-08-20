from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.heroku import HerokuEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Heroku LLM Managed Inference Embedding

The `llama-index-embeddings-heroku` package contains LlamaIndex integrations for building applications with embeddings models on Heroku's Managed Inference platform. This integration allows you to easily connect to and use AI models deployed on Heroku's infrastructure.

## Installation
"""
logger.info("# Heroku LLM Managed Inference Embedding")

# %pip install llama-index-embeddings-heroku

"""
## Setup

### 1. Create a Heroku App

First, create an app in Heroku:

```bash
heroku create $APP_NAME
```

### 2. Create and Attach AI Models

Create and attach a chat model to your app:

```bash
heroku ai:models:create -a $APP_NAME cohere-embed-multilingual --as EMBEDDING
```

### 3. Export Configuration Variables

Export the required configuration variables:

```bash
export EMBEDDING_KEY=$(heroku config:get EMBEDDING_KEY -a $APP_NAME)
export EMBEDDING_MODEL_ID=$(heroku config:get EMBEDDING_MODEL_ID -a $APP_NAME)
export EMBEDDING_URL=$(heroku config:get EMBEDDING_URL -a $APP_NAME)
```

## Usage

### Basic Usage
"""
logger.info("## Setup")


embedding_model = HerokuEmbedding()

embedding = embedding_model.get_text_embedding("Hello, world!")
logger.debug(f"Embedding dimension: {len(embedding)}")

texts = ["Hello", "world", "from", "Heroku"]
embeddings = embedding_model.get_text_embedding_batch(texts)
logger.debug(f"Number of embeddings: {len(embeddings)}")

"""
### Using Environment Variables

The integration automatically reads from environment variables:
"""
logger.info("### Using Environment Variables")


os.environ["EMBEDDING_KEY"] = "your-embedding-key"
os.environ["EMBEDDING_URL"] = "https://us.inference.heroku.com"
os.environ["EMBEDDING_MODEL_ID"] = "claude-3-5-haiku"

llm = HerokuEmbedding()

"""
### Using Parameters

You can also pass parameters directly:
"""
logger.info("### Using Parameters")


embedding_model = HerokuEmbedding(
    model=os.getenv("EMBEDDING_MODEL_ID", "cohere-embed-multilingual"),
    api_key=os.getenv("EMBEDDING_KEY", "your-embedding-key"),
    base_url=os.getenv("EMBEDDING_URL", "https://us.inference.heroku.com"),
    timeout=60.0,
)

logger.debug(embedding_model.get_text_embedding("Hello Heroku!"))

"""
## Available Models

For a complete list of available models, see the [Heroku Managed Inference documentation](https://devcenter.heroku.com/articles/heroku-inference#available-models).

## Error Handling

The integration includes proper error handling for common issues:

- Missing API key
- Invalid inference URL
- Missing model configuration

## Additional Information

For more information about Heroku Managed Inference, visit the [official documentation](https://devcenter.heroku.com/articles/heroku-inference).
"""
logger.info("## Available Models")

logger.info("\n\n[DONE]", bright=True)