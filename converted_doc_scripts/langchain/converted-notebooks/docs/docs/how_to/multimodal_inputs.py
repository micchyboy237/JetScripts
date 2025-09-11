from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from typing import Literal
import base64
import httpx
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
# How to pass multimodal data to models

Here we demonstrate how to pass [multimodal](/docs/concepts/multimodality/) input directly to models.

LangChain supports multimodal data as input to chat models:

1. Following provider-specific formats
2. Adhering to a cross-provider standard

Below, we demonstrate the cross-provider standard. See [chat model integrations](/docs/integrations/chat/) for detail
on native formats for specific providers.

:::note

Most chat models that support multimodal **image** inputs also accept those values in
Ollama's [Chat Completions format](https://platform.ollama.com/docs/guides/images?api-mode=chat):

```python
{
    "type": "image_url",
    "image_url": {"url": image_url},
}
```
:::

## Images

Many providers will accept images passed in-line as base64 data. Some will additionally accept an image from a URL directly.

### Images from base64 data

To pass images in-line, format them as content blocks of the following form:

```python
{
    "type": "image",
    "source_type": "base64",
    "mime_type": "image/jpeg",  # or image/png, etc.
    "data": "<base64 data string>",
}
```

Example:
"""
logger.info("# How to pass multimodal data to models")



image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")


llm = init_chat_model("ollama:llama3.2")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the weather in this image:",
        },
        {
            "type": "image",
            "source_type": "base64",
            "data": image_data,
            "mime_type": "image/jpeg",
        },
    ],
}
response = llm.invoke([message])
logger.debug(response.text())

"""
See [LangSmith trace](https://smith.langchain.com/public/eab05a31-54e8-4fc9-911f-56805da67bef/r) for more detail.

### Images from a URL

Some providers (including [Ollama](/docs/integrations/chat/ollama/),
[Ollama](/docs/integrations/chat/anthropic/), and
[Google Gemini](/docs/integrations/chat/google_generative_ai/)) will also accept images from URLs directly.

To pass images as URLs, format them as content blocks of the following form:

```python
{
    "type": "image",
    "source_type": "url",
    "url": "https://...",
}
```

Example:
"""
logger.info("### Images from a URL")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the weather in this image:",
        },
        {
            "type": "image",
            "source_type": "url",
            "url": image_url,
        },
    ],
}
response = llm.invoke([message])
logger.debug(response.text())

"""
We can also pass in multiple images:
"""
logger.info("We can also pass in multiple images:")

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Are these two images the same?"},
        {"type": "image", "source_type": "url", "url": image_url},
        {"type": "image", "source_type": "url", "url": image_url},
    ],
}
response = llm.invoke([message])
logger.debug(response.text())

"""
## Documents (PDF)

Some providers (including [Ollama](/docs/integrations/chat/ollama/),
[Ollama](/docs/integrations/chat/anthropic/), and
[Google Gemini](/docs/integrations/chat/google_generative_ai/)) will accept PDF documents.

:::note
Ollama requires file-names be specified for PDF inputs. When using LangChain's format, include the `filename` key. See [example below](#example-ollama-file-names).
:::

### Documents from base64 data

To pass documents in-line, format them as content blocks of the following form:

```python
{
    "type": "file",
    "source_type": "base64",
    "mime_type": "application/pdf",
    "data": "<base64 data string>",
}
```

Example:
"""
logger.info("## Documents (PDF)")



pdf_url = "https://pdfobject.com/pdf/sample.pdf"
pdf_data = base64.b64encode(httpx.get(pdf_url).content).decode("utf-8")


llm = init_chat_model("ollama:llama3.2")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the document:",
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": pdf_data,
            "mime_type": "application/pdf",
        },
    ],
}
response = llm.invoke([message])
logger.debug(response.text())

"""
### Documents from a URL

Some providers (specifically [Ollama](/docs/integrations/chat/anthropic/))
will also accept documents from URLs directly.

To pass documents as URLs, format them as content blocks of the following form:

```python
{
    "type": "file",
    "source_type": "url",
    "url": "https://...",
}
```

Example:
"""
logger.info("### Documents from a URL")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the document:",
        },
        {
            "type": "file",
            "source_type": "url",
            "url": pdf_url,
        },
    ],
}
response = llm.invoke([message])
logger.debug(response.text())

"""
## Audio

Some providers (including [Ollama](/docs/integrations/chat/ollama/) and
[Google Gemini](/docs/integrations/chat/google_generative_ai/)) will accept audio inputs.

### Audio from base64 data

To pass audio in-line, format them as content blocks of the following form:

```python
{
    "type": "audio",
    "source_type": "base64",
    "mime_type": "audio/wav",  # or appropriate mime-type
    "data": "<base64 data string>",
}
```

Example:
"""
logger.info("## Audio")



audio_url = "https://upload.wikimedia.org/wikipedia/commons/3/3d/Alcal%C3%A1_de_Henares_%28RPS_13-04-2024%29_canto_de_ruise%C3%B1or_%28Luscinia_megarhynchos%29_en_el_Soto_del_Henares.wav"
audio_data = base64.b64encode(httpx.get(audio_url).content).decode("utf-8")


llm = init_chat_model("google_genai:gemini-2.5-flash")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe this audio:",
        },
        {
            "type": "audio",
            "source_type": "base64",
            "data": audio_data,
            "mime_type": "audio/wav",
        },
    ],
}
response = llm.invoke([message])
logger.debug(response.text())

"""
## Provider-specific parameters

Some providers will support or require additional fields on content blocks containing multimodal data.
For example, Ollama lets you specify [caching](/docs/integrations/chat/anthropic/#prompt-caching) of
specific content to reduce token consumption.

To use these fields, you can:

1. Store them on directly on the content block; or
2. Use the native format supported by each provider (see [chat model integrations](/docs/integrations/chat/) for detail).

We show three examples below.

### Example: Ollama prompt caching
"""
logger.info("## Provider-specific parameters")

llm = init_chat_model("ollama:llama3.2")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the weather in this image:",
        },
        {
            "type": "image",
            "source_type": "url",
            "url": image_url,
            "cache_control": {"type": "ephemeral"},
        },
    ],
}
response = llm.invoke([message])
logger.debug(response.text())
response.usage_metadata

next_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Summarize that in 5 words.",
        }
    ],
}
response = llm.invoke([message, response, next_message])
logger.debug(response.text())
response.usage_metadata

"""
### Example: Ollama citations
"""
logger.info("### Example: Ollama citations")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Generate a 5 word summary of this document.",
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": pdf_data,
            "mime_type": "application/pdf",
            "citations": {"enabled": True},
        },
    ],
}
response = llm.invoke([message])
response.content

"""
### Example: Ollama file names

Ollama requires that PDF documents be associated with file names:
"""
logger.info("### Example: Ollama file names")

llm = init_chat_model("ollama:gpt-4.1")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the document:",
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": pdf_data,
            "mime_type": "application/pdf",
            "filename": "my-file",
        },
    ],
}
response = llm.invoke([message])
logger.debug(response.text())

"""
## Tool calls

Some multimodal models support [tool calling](/docs/concepts/tool_calling) features as well. To call tools using such models, simply bind tools to them in the [usual way](/docs/how_to/tool_calling), and invoke the model using content blocks of the desired type (e.g., containing image data).
"""
logger.info("## Tool calls")




@tool
def weather_tool(weather: Literal["sunny", "cloudy", "rainy"]) -> None:
    """Describe the weather"""
    pass


llm_with_tools = llm.bind_tools([weather_tool])

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the weather in this image:"},
        {"type": "image", "source_type": "url", "url": image_url},
    ],
}
response = llm_with_tools.invoke([message])
response.tool_calls

logger.info("\n\n[DONE]", bright=True)