from datasets import load_dataset
from haystack import Document
from haystack import Pipeline
from haystack import component
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.generators import OllamaFunctionCallingAdapterGenerator
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.component.types import Variadic
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from itertools import chain
from jet.logger import CustomLogger
from typing import Any
from typing import List
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# 🗣️ Conversational RAG using Memory

In this notebook, we'll explore how to incorporate memory into a RAG pipeline to enable conversations with our documents, using an `InMemoryChatMessageStore`, a `ChatMessageRetriever`, and a `ChatMessageWriter`.

**Useful Sources**

* [📖 Docs](https://docs.haystack.deepset.ai/docs/intro)
* [📚 Tutorials](https://haystack.deepset.ai/tutorials)

## Installation

Install Haystack, `haystack-experimental` and `datasets` with pip:
"""
logger.info("# 🗣️ Conversational RAG using Memory")

# !pip install -U haystack-ai datasets

"""
## Enter OllamaFunctionCalling API key
"""
logger.info("## Enter OllamaFunctionCalling API key")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter OllamaFunctionCalling API key:")

"""
## Create DocumentStore and Index Documents

Create an index with [seven-wonders](https://huggingface.co/datasets/bilgeyucel/seven-wonders) dataset:
"""
logger.info("## Create DocumentStore and Index Documents")


dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

document_store = InMemoryDocumentStore()
document_store.write_documents(documents=docs)

"""
## Create Memory

Memory, so the conversation history, is saved as `ChatMessage` objects in a `InMemoryChatMessageStore`. When required, you can retrieve the conversation history from the chat message store using `ChatMessageRetriever`.

To store memory, initialize an `InMemoryChatMessageStore`, a `ChatMessageRetriever` and a `ChatMessageWriter`. Import these components from the [`haystack-experimental`](https://github.com/deepset-ai/haystack-experimental) package:
"""
logger.info("## Create Memory")


memory_store = InMemoryChatMessageStore()
memory_retriever = ChatMessageRetriever(memory_store)
memory_writer = ChatMessageWriter(memory_store)

"""
## Prompt Template for RAG with Memory

Prepare a prompt template for RAG and additionally, add another section for memory. Memory info will be retrieved by `ChatMessageRetriever` from the `InMemoryChatMessageStore` and injected into the prompt through `memories` prompt variable.
"""
logger.info("## Prompt Template for RAG with Memory")


system_message = ChatMessage.from_system("You are a helpful AI assistant using provided supporting documents and conversation history to assist humans")

user_message_template ="""Given the conversation history and the provided supporting documents, give a brief answer to the question.
Note that supporting documents are not part of the conversation. If question can't be answered from supporting documents, say so.

    Conversation history:
    {% for memory in memories %}
        {{ memory.text }}
    {% endfor %}

    Supporting documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{query}}
    \nAnswer:
"""
user_message = ChatMessage.from_user(user_message_template)

"""
## Build the Pipeline

Add components for RAG and memory to build your pipeline. Incorporate the custom `ListJoiner` component into your pipeline to handle messages from both the user and the LLM, writing them to the memory store.

> **Note**: The `ListJoiner` component will be available in Haystack starting from version 2.8.0!
"""
logger.info("## Build the Pipeline")




@component
class ListJoiner:
    def __init__(self, _type: Any):
        component.set_output_types(self, values=_type)

    def run(self, values: Variadic[Any]):
        result = list(chain(*values))
        return {"values": result}


pipeline = Pipeline()

pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=3))
pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "memories"], required_variables=["query", "documents", "memories"]))
pipeline.add_component("llm", OllamaFunctionCallingAdapterChatGenerator())

pipeline.add_component("memory_retriever", memory_retriever)
pipeline.add_component("memory_writer", memory_writer)
pipeline.add_component("memory_joiner", ListJoiner(List[ChatMessage]))

pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.messages")
pipeline.connect("llm.replies", "memory_joiner")

pipeline.connect("memory_joiner", "memory_writer")
pipeline.connect("memory_retriever", "prompt_builder.memories")

"""
### Visualize the pipeline

Visualize the pipeline with the [`show()`](https://docs.haystack.deepset.ai/docs/visualizing-pipelines) method to confirm the connections are correct.
"""
logger.info("### Visualize the pipeline")

pipeline.show()

"""
## Run the Pipeline

Test the pipeline with some queries. Ensure that every user query is also sent to the `memory_joiner` so that both the user queries and the LLM responses are stored together in the memory store.

Here are example queries you can try:

* *What does Rhodes Statue look like?*
* *Who built it?*
"""
logger.info("## Run the Pipeline")

while True:
    messages = [system_message, user_message]
    question = input("Enter your question or Q to exit.\n🧑 ")
    if question=="Q":
        break

    res = pipeline.run(data={"retriever": {"query": question},
                             "prompt_builder": {"template": messages, "query": question},
                             "memory_joiner": {"values": [ChatMessage.from_user(question)]}},
                            include_outputs_from=["llm"])
    assistant_resp = res['llm']['replies'][0]
    logger.debug(f"🤖 {assistant_resp.content}")

"""
⚠️ If you followed the example queries, you'll notice that the second question was answered incorrectly. This happened because the retrieved documents weren't relevant to the user's query. The retrieval was based on the query "*Who built it?*", which doesn't have enough context to retrieve documents. Let's fix it with **rephrasing the query for search**.

## Prompt Template for Rephrasing User Query

In conversational systems, simply injecting memory into the prompt is not enough to perform RAG effectively. There needs to be a mechanism to rephrase the user's query based on the conversation history to ensure relevant documents are retrieved. For instance, if the first user query is "*What's the first name of Einstein?*" and the second query is "*Where was he born?*", the system should understand that "he" refers to Einstein. The rephrasing mechanism should then modify the second query to "*Where was Einstein born?*" to retrieve the correct documents.

We can use an LLM to rephrase the user's query. Let's create a prompt that instructs the LLM to rephrase the query, incorporating the conversation history, to make it suitable for retrieving relevant documents.
"""
logger.info("## Prompt Template for Rephrasing User Query")

query_rephrase_template = """
        Rewrite the question for search while keeping its meaning and key terms intact.
        If the conversation history is empty, DO NOT change the query.
        Use conversation history only if necessary, and avoid extending the query with your own knowledge.
        If no changes are needed, output the current question as is.

        Conversation history:
        {% for memory in memories %}
            {{ memory.text }}
        {% endfor %}

        User Query: {{query}}
        Rewritten Query:
"""

"""
## Build the Conversational RAG Pipeline

Now, let's incorporate query rephrasing into our pipeline by adding a new [PromptBuilder](https://docs.haystack.deepset.ai/docs/promptbuilder) with the prompt above, [OllamaFunctionCallingAdapterGenerator](https://docs.haystack.deepset.ai/docs/openaigenerator), and an [OutputAdapter](https://docs.haystack.deepset.ai/docs/outputadapter). The `OllamaFunctionCallingAdapterGenerator` will rephrase the user's query for search, and the `OutputAdapter` will convert the output from the `OllamaFunctionCallingAdapterGenerator` into the input for the `InMemoryBM25Retriever`. The rest of the pipeline will be the same.
"""
logger.info("## Build the Conversational RAG Pipeline")


conversational_rag = Pipeline()

conversational_rag.add_component("query_rephrase_prompt_builder", PromptBuilder(query_rephrase_template))
conversational_rag.add_component("query_rephrase_llm", OllamaFunctionCallingAdapterGenerator())
conversational_rag.add_component("list_to_str_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))

conversational_rag.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=3))
conversational_rag.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "memories"], required_variables=["query", "documents", "memories"]))
conversational_rag.add_component("llm", OllamaFunctionCallingAdapterChatGenerator())

conversational_rag.add_component("memory_retriever", ChatMessageRetriever(memory_store))
conversational_rag.add_component("memory_writer", ChatMessageWriter(memory_store))
conversational_rag.add_component("memory_joiner", ListJoiner(List[ChatMessage]))

conversational_rag.connect("memory_retriever", "query_rephrase_prompt_builder.memories")
conversational_rag.connect("query_rephrase_prompt_builder.prompt", "query_rephrase_llm")
conversational_rag.connect("query_rephrase_llm.replies", "list_to_str_adapter")
conversational_rag.connect("list_to_str_adapter", "retriever.query")

conversational_rag.connect("retriever.documents", "prompt_builder.documents")
conversational_rag.connect("prompt_builder.prompt", "llm.messages")
conversational_rag.connect("llm.replies", "memory_joiner")

conversational_rag.connect("memory_joiner", "memory_writer")
conversational_rag.connect("memory_retriever", "prompt_builder.memories")

"""
## Let's have a conversation 😀

Now, run the pipeline with the relevant inputs. Instead of sending the query directly to the `retriever`, this time, pass it to the `query_rephrase_prompt_builder` to rephrase it.

Here are some example queries and follow ups you can try:

* *What does Rhodes Statue look like?* - *Who built it?* - *Did he destroy it?*
* *Where is Gardens of Babylon?* - *When was it built?*
"""
logger.info("## Let's have a conversation 😀")

while True:
    messages = [system_message, user_message]
    question = input("Enter your question or Q to exit.\n🧑 ")
    if question=="Q":
        break

    res = conversational_rag.run(data={"query_rephrase_prompt_builder": {"query": question},
                             "prompt_builder": {"template": messages, "query": question},
                             "memory_joiner": {"values": [ChatMessage.from_user(question)]}},
                            include_outputs_from=["llm","query_rephrase_llm"])
    search_query = res['query_rephrase_llm']['replies'][0]
    logger.debug(f"   🔎 Search Query: {search_query}")
    assistant_resp = res['llm']['replies'][0]
    logger.debug(f"🤖 {assistant_resp.text}")

"""
✅ Notice that this time, with the help of query rephrasing, we've built a conversational RAG pipeline that can handle follow-up queries and retrieve the relevant documents.
"""

logger.info("\n\n[DONE]", bright=True)