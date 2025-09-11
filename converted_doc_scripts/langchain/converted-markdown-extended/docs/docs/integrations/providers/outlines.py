from jet.logger import logger
from langchain_community.chat_models import ChatOutlines
from langchain_community.llms import Outlines
from langchain_core.messages import HumanMessage
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
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
# Outlines

>[Outlines](https://github.com/dottxt-ai/outlines) is a Python library for constrained language generation. It provides a unified interface to various language models and allows for structured generation using techniques like regex matching, type constraints, JSON schemas, and context-free grammars.

Outlines supports multiple backends, including:
- Hugging Face Transformers
- llama.cpp
- vLLM
- MLX

This integration allows you to use Outlines models with LangChain, providing both LLM and chat model interfaces.

## Installation and Setup

To use Outlines with LangChain, you'll need to install the Outlines library:
"""
logger.info("# Outlines")

pip install outlines

"""
Depending on the backend you choose, you may need to install additional dependencies:

- For Transformers: `pip install transformers torch datasets`
- For llama.cpp: `pip install llama-cpp-python`
- For vLLM: `pip install vllm`
- For MLX: `pip install mlx`

## LLM

To use Outlines as an LLM in LangChain, you can use the `Outlines` class:
"""
logger.info("## LLM")


"""
## Chat Models

To use Outlines as a chat model in LangChain, you can use the `ChatOutlines` class:
"""
logger.info("## Chat Models")


"""
## Model Configuration

Both `Outlines` and `ChatOutlines` classes share similar configuration options:
"""
logger.info("## Model Configuration")

model = Outlines(
    model="meta-llama/Llama-2-7b-chat-hf",  # Model identifier
    backend="transformers",  # Backend to use (transformers, llamacpp, vllm, or mlxlm)
    max_tokens=256,  # Maximum number of tokens to generate
    stop=["\n"],  # Optional list of stop strings
    streaming=True,  # Whether to stream the output
    regex=None,
    type_constraints=None,
    json_schema=None,
    grammar=None,
    model_kwargs={"temperature": 0.7}
)

"""
### Model Identifier

The `model` parameter can be:
- A Hugging Face model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
- A local path to a model
- For GGUF models, the format is "repo_id/file_name" (e.g., "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf")

### Backend Options

The `backend` parameter specifies which backend to use:
- `"transformers"`: For Hugging Face Transformers models (default)
- `"llamacpp"`: For GGUF models using llama.cpp
- `"transformers_vision"`: For vision-language models (e.g., LLaVA)
- `"vllm"`: For models using the vLLM library
- `"mlxlm"`: For models using the MLX framework

### Structured Generation

Outlines provides several methods for structured generation:

1. **Regex Matching**:
   ```python
   model = Outlines(
       model="meta-llama/Llama-2-7b-chat-hf",
       regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
   )
   ```
   This will ensure the generated text matches the specified regex pattern (in this case, a valid IP address).

2. **Type Constraints**:
   ```python
   model = Outlines(
       model="meta-llama/Llama-2-7b-chat-hf",
       type_constraints=int
   )
   ```
   This restricts the output to valid Python types (int, float, bool, datetime.date, datetime.time, datetime.datetime).

3. **JSON Schema**:
   ```python

   class Person(BaseModel):
       name: str
       age: int

   model = Outlines(
       model="meta-llama/Llama-2-7b-chat-hf",
       json_schema=Person
   )
   ```
   This ensures the generated output adheres to the specified JSON schema or Pydantic model.

4. **Context-Free Grammar**:
   ```python
   model = Outlines(
       model="meta-llama/Llama-2-7b-chat-hf",
       grammar="""
logger.info("### Model Identifier")
           ?start: expression
           ?expression: term (("+" | "-") term)*
           ?term: factor (("*" | "/") factor)*
           ?factor: NUMBER | "-" factor | "(" expression ")"
           %import common.NUMBER
       """
   )
   ```
   This generates text that adheres to the specified context-free grammar in EBNF format.

## Usage Examples

### LLM Example
"""
logger.info("## Usage Examples")


llm = Outlines(model="meta-llama/Llama-2-7b-chat-hf", max_tokens=100)
result = llm.invoke("Tell me a short story about a robot.")
logger.debug(result)

"""
### Chat Model Example
"""
logger.info("### Chat Model Example")


chat = ChatOutlines(model="meta-llama/Llama-2-7b-chat-hf", max_tokens=100)
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What's the capital of France?")
]
result = chat.invoke(messages)
logger.debug(result.content)

"""
### Streaming Example
"""
logger.info("### Streaming Example")


chat = ChatOutlines(model="meta-llama/Llama-2-7b-chat-hf", streaming=True)
for chunk in chat.stream("Tell me a joke about programming."):
    logger.debug(chunk.content, end="", flush=True)
logger.debug()

"""
### Structured Output Example
"""
logger.info("### Structured Output Example")


class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

llm = Outlines(
    model="meta-llama/Llama-2-7b-chat-hf",
    json_schema=MovieReview
)
result = llm.invoke("Write a short review for the movie 'Inception'.")
logger.debug(result)

"""
## Additional Features

### Tokenizer Access

You can access the underlying tokenizer for the model:
"""
logger.info("## Additional Features")

tokenizer = llm.tokenizer
encoded = tokenizer.encode("Hello, world!")
decoded = tokenizer.decode(encoded)

logger.info("\n\n[DONE]", bright=True)