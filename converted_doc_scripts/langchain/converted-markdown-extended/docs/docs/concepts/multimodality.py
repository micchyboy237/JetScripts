from jet.logger import logger
from langchain_core.messages import HumanMessage
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Multimodality

## Overview

**Multimodality** refers to the ability to work with data that comes in different forms, such as text, audio, images, and video. Multimodality can appear in various components, allowing models and systems to handle and process a mix of these data types seamlessly.

- **Chat Models**: These could, in theory, accept and generate multimodal inputs and outputs, handling a variety of data types like text, images, audio, and video.
- **Embedding Models**: Embedding Models can represent multimodal content, embedding various forms of data—such as text, images, and audio—into vector spaces.
- **Vector Stores**: Vector stores could search over embeddings that represent multimodal data, enabling retrieval across different types of information.

## Multimodality in chat models

:::info Pre-requisites
* [Chat models](/docs/concepts/chat_models)
* [Messages](/docs/concepts/messages)
:::

LangChain supports multimodal data as input to chat models:

1. Following provider-specific formats
2. Adhering to a cross-provider standard (see [how-to guides](/docs/how_to/#multimodal) for detail)

### How to use multimodal models

* Use the [chat model integration table](/docs/integrations/chat/) to identify which models support multimodality.
* Reference the [relevant how-to guides](/docs/how_to/#multimodal) for specific examples of how to use multimodal models.

### What kind of multimodality is supported?

#### Inputs

Some models can accept multimodal inputs, such as images, audio, video, or files.
The types of multimodal inputs supported depend on the model provider. For instance,
[Ollama](/docs/integrations/chat/ollama/),
[Ollama](/docs/integrations/chat/anthropic/), and
[Google Gemini](/docs/integrations/chat/google_generative_ai/)
support documents like PDFs as inputs.

The gist of passing multimodal inputs to a chat model is to use content blocks that
specify a type and corresponding data. For example, to pass an image to a chat model
as URL:
"""
logger.info("# Multimodality")


message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        {
            "type": "image",
            "source_type": "url",
            "url": "https://...",
        },
    ],
)
response = model.invoke([message])

"""
We can also pass the image as in-line data:
"""
logger.info("We can also pass the image as in-line data:")


message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        {
            "type": "image",
            "source_type": "base64",
            "data": "<base64 string>",
            "mime_type": "image/jpeg",
        },
    ],
)
response = model.invoke([message])

"""
To pass a PDF file as in-line data (or URL, as supported by providers such as
Ollama), just change `"type"` to `"file"` and `"mime_type"` to `"application/pdf"`.

See the [how-to guides](/docs/how_to/#multimodal) for more detail.

Most chat models that support multimodal **image** inputs also accept those values in
Ollama's [Chat Completions format](https://platform.ollama.com/docs/guides/images?api-mode=chat):
"""
logger.info("To pass a PDF file as in-line data (or URL, as supported by providers such as")


message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ],
)
response = model.invoke([message])

"""
Otherwise, chat models will typically accept the native, provider-specific content
block format. See [chat model integrations](/docs/integrations/chat/) for detail
on specific providers.


#### Outputs

Some chat models support multimodal outputs, such as images and audio. Multimodal
outputs will appear as part of the [AIMessage](/docs/concepts/messages/#aimessage)
response object. See for example:

- Generating [audio outputs](/docs/integrations/chat/ollama/#audio-generation-preview) with Ollama;
- Generating [image outputs](/docs/integrations/chat/google_generative_ai/#multimodal-usage) with Google Gemini.

#### Tools

Currently, no chat model is designed to work **directly** with multimodal data in a [tool call request](/docs/concepts/tool_calling) or [ToolMessage](/docs/concepts/tool_calling) result.

However, a chat model can easily interact with multimodal data by invoking tools with references (e.g., a URL) to the multimodal data, rather than the data itself. For example, any model capable of [tool calling](/docs/concepts/tool_calling) can be equipped with tools to download and process images, audio, or video.

## Multimodality in embedding models

:::info Prerequisites
* [Embedding Models](/docs/concepts/embedding_models)
:::

**Embeddings** are vector representations of data used for tasks like similarity search and retrieval.

The current [embedding interface](https://python.langchain.com/api_reference/core/embeddings/langchain_core.embeddings.embeddings.Embeddings.html#langchain_core.embeddings.embeddings.Embeddings) used in LangChain is optimized entirely for text-based data, and will **not** work with multimodal data.

As use cases involving multimodal search and retrieval tasks become more common, we expect to expand the embedding interface to accommodate other data types like images, audio, and video.

## Multimodality in vector stores

:::info Prerequisites
* [Vector stores](/docs/concepts/vectorstores)
:::

Vector stores are databases for storing and retrieving embeddings, which are typically used in search and retrieval tasks. Similar to embeddings, vector stores are currently optimized for text-based data.

As use cases involving multimodal search and retrieval tasks become more common, we expect to expand the vector store interface to accommodate other data types like images, audio, and video.
"""
logger.info("#### Outputs")

logger.info("\n\n[DONE]", bright=True)