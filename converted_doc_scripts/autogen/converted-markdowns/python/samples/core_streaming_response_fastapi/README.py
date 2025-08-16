from jet.logger import CustomLogger
import json
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AutoGen-Core Streaming Chat API with FastAPI

This sample demonstrates how to build a streaming chat API with multi-turn conversation history using `autogen-core` and FastAPI.

## Key Features

1.  **Streaming Response**: Implements real-time streaming of LLM responses by utilizing FastAPI's `StreamingResponse`, `autogen-core`'s asynchronous features, and a global queue created with `asyncio.Queue()` to manage the data stream, thereby providing faster user-perceived response times.
2.  **Multi-Turn Conversation**: The Agent (`MyAgent`) can receive and process chat history records (`ChatHistory`) containing multiple turns of interaction, enabling context-aware continuous conversations.

## File Structure

*   `app.py`: FastAPI application code, including API endpoints, Agent definitions, runtime settings, and streaming logic.
*   `README.md`: (This document) Project introduction and usage instructions.

## Installation

First, make sure you have Python installed (recommended 3.8 or higher). Then, in your project directory, install the necessary libraries via pip:
"""
logger.info("# AutoGen-Core Streaming Chat API with FastAPI")

pip install "fastapi" "uvicorn[standard]" "autogen-core" "autogen-ext[openai]"

"""
## Configuration

Create a new file named `model_config.yaml` in the same directory as this README file to configure your model settings.
See `model_config_template.yaml` for an example.

**Note**: Hardcoding API keys directly in the code is only suitable for local testing. For production environments, it is strongly recommended to use environment variables or other secure methods to manage keys.

## Running the Application

In the directory containing `app.py`, run the following command to start the FastAPI application:
"""
logger.info("## Configuration")

uvicorn app:app --host 0.0.0.0 --port 8501 --reload

"""
After the service starts, the API endpoint will be available at `http://<your-server-ip>:8501/chat/completions`.

## Using the API

You can interact with the Agent by sending a POST request to the `/chat/completions` endpoint. The request body must be in JSON format and contain a `messages` field, the value of which is a list, where each element represents a turn of conversation.

**Request Body Format**:
"""
logger.info("## Using the API")

{
  "messages": [
    {"source": "user", "content": "Hello!"},
    {"source": "assistant", "content": "Hello! How can I help you?"},
    {"source": "user", "content": "Introduce yourself."}
  ]
}

"""
**Example (using curl)**:
"""

curl -N -X POST http://localhost:8501/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"source": "user", "content": "Hello, I'\''m Tory."},
    {"source": "assistant", "content": "Hello Tory, nice to meet you!"},
    {"source": "user", "content": "Say hello by my name and introduce yourself."}
  ]
}'

"""
**Example (using Python requests)**:
"""

url = "http://localhost:8501/chat/completions"
data = {
    'stream': True,
    'messages': [
            {'source': 'user', 'content': "Hello,I'm tory."},
            {'source': 'assistant', 'content':"hello Tory, nice to meet you!"},
            {'source': 'user', 'content': "Say hello by my name and introduce yourself."}
        ]
    }
headers = {'Content-Type': 'application/json'}
try:
    response = requests.post(url, json=data, headers=headers, stream=True)
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            logger.debug(json.loads(chunk)["content"], end='', flush=True)

except requests.exceptions.RequestException as e:
    logger.debug(f"Error: {e}")
except json.JSONDecodeError as e:
    logger.debug(f"JSON Decode Error: {e}")

logger.info("\n\n[DONE]", bright=True)