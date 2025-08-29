# import tiktoken
import json
from transformers import AutoTokenizer
from jet.llm.ollama.base import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
import os
from jet.token import tokenizer
from jet.transformers.object import make_serializable
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings, OLLAMA_HF_MODELS
initialize_ollama_settings()

# Chat Summary Memory Buffer
# In this demo, we use the new *ChatSummaryMemoryBuffer* to limit the chat history to a certain token length, and iteratively summarize all messages that do not fit in the memory buffer. This can be useful if you want to limit costs and latency (assuming the summarization prompt uses and generates fewer tokens than including the entire history).
#
# The original *ChatMemoryBuffer* gives you the option to truncate the history after a certain number of tokens, which is useful to limit costs and latency, but also removes potentially relevant information from the chat history.
#
# The newer *ChatSummaryMemoryBuffer* aims to makes this a bit more flexible, so the user has more control over which chat_history is retained.

# %pip install llama-index-llms-ollama
# %pip install llama-index


# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


# First, we simulate some chat history that will not fit in the memory buffer in its entirety.

chat_history = [
    ChatMessage(role="user", content="What is LlamaIndex?"),
    ChatMessage(
        role="assistant",
        content="LlamaaIndex is the leading data framework for building LLM applications",
    ),
    ChatMessage(role="user", content="Can you give me some more details?"),
    ChatMessage(
        role="assistant",
        content="""LlamaIndex is a framework for building context-augmented LLM applications. Context augmentation refers to any use case that applies LLMs on top of your private or domain-specific data. Some popular use cases include the following: 
        Question-Answering Chatbots (commonly referred to as RAG systems, which stands for "Retrieval-Augmented Generation"), Document Understanding and Extraction, Autonomous Agents that can perform research and take actions
        LlamaIndex provides the tools to build any of these above use cases from prototype to production. The tools allow you to both ingest/process this data and implement complex query workflows combining data access with LLM prompting.""",
    ),
]

# By supplying an *llm* and *token_limit* for summarization, we create a *ChatSummaryMemoryBuffer* instance.

summarizer_model = "mistral"
summarizer_llm = Ollama(
    model=summarizer_model)
# tokenizer_fn = tiktoken.encoding_for_model(summarizer_llm.model).encode
# tokenizer_fn = AutoTokenizer.from_pretrained(
#     OLLAMA_HF_MODELS[summarizer_model]).encode
tokenizer_fn = tokenizer().encode
memory = ChatSummaryMemoryBuffer.from_defaults(
    chat_history=chat_history,
    llm=summarizer_llm,
    token_limit=2,
    tokenizer_fn=tokenizer_fn,
)

history = memory.get()

# When printing the history, we can observe that older messages have been summarized.
logger.newline()
logger.info("History result 1:")
logger.success(json.dumps(make_serializable(history), indent=2))

# Let's add some new chat history.

new_chat_history = [
    ChatMessage(role="user", content="Why context augmentation?"),
    ChatMessage(
        role="assistant",
        content="LLMs offer a natural language interface between humans and data. Widely available models come pre-trained on huge amounts of publicly available data. However, they are not trained on your data, which may be private or specific to the problem you're trying to solve. It's behind APIs, in SQL databases, or trapped in PDFs and slide decks. LlamaIndex provides tooling to enable context augmentation. A popular example is Retrieval-Augmented Generation (RAG) which combines context with LLMs at inference time. Another is finetuning.",
    ),
    ChatMessage(role="user", content="Who is LlamaIndex for?"),
    ChatMessage(
        role="assistant",
        content="LlamaIndex provides tools for beginners, advanced users, and everyone in between. Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in 5 lines of code. For more complex applications, our lower-level APIs allow advanced users to customize and extend any module—data connectors, indices, retrievers, query engines, reranking modules—to fit their needs.",
    ),
]
memory.put(new_chat_history[0])
memory.put(new_chat_history[1])
memory.put(new_chat_history[2])
memory.put(new_chat_history[3])
history = memory.get()

# The history will now be updated with a new summary, containing the latest information.
logger.newline()
logger.info("History result 2:")
logger.success(json.dumps(make_serializable(history), indent=2))

# Using a longer *token_limit* allows the user to control the balance between retaining the full chat history and summarization.

memory = ChatSummaryMemoryBuffer.from_defaults(
    chat_history=chat_history + new_chat_history,
    llm=summarizer_llm,
    token_limit=256,
    tokenizer_fn=tokenizer_fn,
)
memory_result = memory.get()

logger.newline()
logger.info("Memory result:")
logger.success(json.dumps(make_serializable(memory_result), indent=2))

logger.info("\n\n[DONE]", bright=True)
