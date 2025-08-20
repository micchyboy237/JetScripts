from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.heroku import Heroku
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
# Heroku LLM Managed Inference

The `llama-index-llms-heroku` package contains LlamaIndex integrations for building applications with models on Heroku's Managed Inference platform. This integration allows you to easily connect to and use AI models deployed on Heroku's infrastructure.

## Installation
"""
logger.info("# Heroku LLM Managed Inference")

# %pip install llama-index-llms-heroku

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
heroku ai:models:create -a $APP_NAME claude-3-5-haiku
```

### 3. Export Configuration Variables

Export the required configuration variables:

```bash
export INFERENCE_KEY=$(heroku config:get INFERENCE_KEY -a $APP_NAME)
export INFERENCE_MODEL_ID=$(heroku config:get INFERENCE_MODEL_ID -a $APP_NAME)
export INFERENCE_URL=$(heroku config:get INFERENCE_URL -a $APP_NAME)
```

## Usage

### Basic Usage
"""
logger.info("## Setup")


llm = Heroku()

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="What are the most popular house pets in North America?",
    ),
]

response = llm.chat(messages)
logger.debug(response)

"""
### Using Environment Variables

The integration automatically reads from environment variables:
"""
logger.info("### Using Environment Variables")


os.environ["INFERENCE_KEY"] = "your-inference-key"
os.environ["INFERENCE_URL"] = "https://us.inference.heroku.com"
os.environ["INFERENCE_MODEL_ID"] = "claude-3-5-haiku"

llm = Heroku()

"""
### Using Parameters

You can also pass parameters directly:
"""
logger.info("### Using Parameters")


llm = Heroku(
    model=os.getenv("INFERENCE_MODEL_ID", "claude-3-5-haiku"),
    api_key=os.getenv("INFERENCE_KEY", "your-inference-key"),
    inference_url=os.getenv(
        "INFERENCE_URL", "https://us.inference.heroku.com"
    ),
    max_tokens=1024,
)

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