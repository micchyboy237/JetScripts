from google.oauth2 import service_account
from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.vertex import Vertex
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Vertex AI

**NOTE:** Vertex has largely been replaced by Google GenAI, which supports the same functionality from Vertex using the `google-genai` package. Visit the [Google GenAI page](https://docs.llamaindex.ai/en/stable/examples/llm/google_genai/) for the latest examples and documentation.

## Installing Vertex AI 
To Install Vertex AI you need to follow the following steps
* Install Vertex Cloud SDK (https://googleapis.dev/python/aiplatform/latest/index.html)
* Setup your Default Project, credentials, region
# Basic auth example for service account
"""
logger.info("# Vertex AI")

# %pip install llama-index-llms-vertex


filename = "vertex-407108-37495ce6c303.json"
credentials: service_account.Credentials = (
    service_account.Credentials.from_service_account_file(filename)
)
Vertex(
    model="text-bison", project=credentials.project_id, credentials=credentials
)

"""
## Basic Usage
Basic call to the text-bison model
"""
logger.info("## Basic Usage")


llm = Vertex(model="text-bison", temperature=0, additional_kwargs={})
llm.complete("Hello this is a sample text").text

"""
## Async Usage
### Async
"""
logger.info("## Async Usage")

(llm.complete("hello")).text

"""
# Streaming Usage 
### Streaming
"""
logger.info("# Streaming Usage")

list(llm.stream_complete("hello"))[-1].text

"""
# Chat Usage
### chat generation
"""
logger.info("# Chat Usage")

chat = Vertex(model="chat-bison")
messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="Reply everything in french"),
    ChatMessage(role=MessageRole.USER, content="Hello"),
]

chat.chat(messages=messages).message.content

"""
# Async Chat
### Asynchronous chat response
"""
logger.info("# Async Chat")

(chat.chat(messages=messages)).message.content

"""
# Streaming Chat
### streaming chat response
"""
logger.info("# Streaming Chat")

list(chat.stream_chat(messages=messages))[-1].message.content

"""
# Gemini Models
Calling Google Gemini Models using Vertex AI is fully supported.
### Gemini Pro
"""
logger.info("# Gemini Models")

llm = Vertex(
    model="gemini-pro",
    project=credentials.project_id,
    credentials=credentials,
    context_window=100000,
)
llm.complete("Hello Gemini").text

"""
### Gemini Vision Models
Gemini vision-capable models now support `TextBlock` and `ImageBlock` for structured multi-modal inputs, replacing the older dictionary-based `content` format. Use `blocks` to include text and images via file paths or URLs.

**Example with Image Path:**
"""
logger.info("### Gemini Vision Models")


history = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="sample.jpg"),
            TextBlock(text="What is in this image?"),
        ],
    ),
]
llm = Vertex(
    model="gemini-1.5-flash",
    project=credentials.project_id,
    credentials=credentials,
    context_window=100000,
)
logger.debug(llm.chat(history).message.content)

"""
**Example with Image URL:**
"""


history = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(
                url="https://upload.wikimedia.org/wikipedia/commons/7/71/Sibirischer_tiger_de_edit02.jpg"
            ),
            TextBlock(text="What is in this image?"),
        ],
    ),
]
llm = Vertex(
    model="gemini-1.5-flash",
    project=credentials.project_id,
    credentials=credentials,
    context_window=100000,
)
logger.debug(llm.chat(history).message.content)

logger.info("\n\n[DONE]", bright=True)