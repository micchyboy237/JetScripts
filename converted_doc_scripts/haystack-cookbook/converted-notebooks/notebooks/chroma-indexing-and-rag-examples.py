from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses.chat_message import ChatMessage
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from jet.logger import CustomLogger
from pathlib import Path
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
# Use ChromaDocumentStore with Haystack

## Install dependencies
"""
logger.info("# Use ChromaDocumentStore with Haystack")

# !pip install -U chroma-haystack "huggingface_hub>=0.22.0"

"""
## Indexing Pipeline: preprocess, split and index documents
In this section, we will index documents into a Chroma DB collection by building a Haystack indexing pipeline. Here, we are indexing documents from the [VIM User Manuel](https://vimhelp.org/) into the Haystack [`ChromaDocumentStore`](https://haystack.deepset.ai/integrations/chroma-documentstore).

 We have the `.txt` files for these pages in the examples folder for the `ChromaDocumentStore`, so we are using the [`TextFileToDocument`](https://docs.haystack.deepset.ai/docs/textfiletodocument) and [`DocumentWriter`](https://docs.haystack.deepset.ai/docs/documentwriter) components to build this indexing pipeline.
"""
logger.info("## Indexing Pipeline: preprocess, split and index documents")

# !curl -sL https://github.com/deepset-ai/haystack-core-integrations/tarball/main -o main.tar
# !mkdir main
# !tar xf main.tar -C main --strip-components 1
# !mv main/integrations/chroma/example/data .




file_paths = ["data" / Path(name) for name in os.listdir("data")]

document_store = ChromaDocumentStore()

indexing = Pipeline()
indexing.add_component("converter", TextFileToDocument())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "writer")
indexing.run({"converter": {"sources": file_paths}})

"""
## Query Pipeline: build retrieval-augmented generation (RAG) pipelines

Once we have documents in the `ChromaDocumentStore`, we can use the accompanying Chroma retrievers to build a query pipeline. The query pipeline below is a simple retrieval-augmented generation (RAG) pipeline that uses Chroma's [query API](https://docs.trychroma.com/usage-guide#querying-a-collection).

You can change the indexing pipeline and query pipelines here for embedding search by using one of the [`Haystack Embedders`](https://docs.haystack.deepset.ai/docs/embedders) accompanied by the  `ChromaEmbeddingRetriever`.


In this example we are using:
- The [`OllamaFunctionCallingAdapterChatGenerator`](https://docs.haystack.deepset.ai/docs/openaichatgenerator) with `llama3.2`. (You will need a OllamaFunctionCallingAdapter API key to use this model). You can replace this with any of the other [`Generators`](https://docs.haystack.deepset.ai/docs/generators)
- The [`ChatPromptBuilder`](https://docs.haystack.deepset.ai/docs/chatpromptbuilder) which holds the prompt template. You can adjust this to a prompt of your choice
- The [`ChromaQueryTextRetriver`](https://docs.haystack.deepset.ai/docs/chromaqueryretriever) which expects a list of queries and retieves the `top_k` most relevant documents from your Chroma collection.
"""
logger.info("## Query Pipeline: build retrieval-augmented generation (RAG) pipelines")

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = getpass("Enter OllamaFunctionCallingAdapter API key:")


prompt = """
Answer the query based on the provided context.
If the context does not contain the answer, say 'Answer not found'.
Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
query: {{query}}
Answer:
"""

template = [ChatMessage.from_user(prompt)]
prompt_builder = ChatPromptBuilder(template=template)

llm = OllamaFunctionCallingAdapterChatGenerator()
retriever = ChromaQueryTextRetriever(document_store)

querying = Pipeline()
querying.add_component("retriever", retriever)
querying.add_component("prompt_builder", prompt_builder)
querying.add_component("llm", llm)

querying.connect("retriever.documents", "prompt_builder.documents")
querying.connect("prompt_builder", "llm")

query = "Should I write documentation for my plugin?"
results = querying.run({"retriever": {"query": query, "top_k": 3}, "prompt_builder": {"query": query}})

logger.debug(results["llm"]["replies"][0].text)

logger.info("\n\n[DONE]", bright=True)