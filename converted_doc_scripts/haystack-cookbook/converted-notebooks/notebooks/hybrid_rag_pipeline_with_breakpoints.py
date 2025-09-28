from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
# from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.core.errors import BreakpointException
from jet.adapters.haystack.ollama_chat_generator import OllamaChatGenerator
from jet.file.utils import save_file
from jet.logger import logger
import glob
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# Hybrid RAG Pipeline with Breakpoints

This notebook demonstrates how to setup breakpoints in a Haystack pipeline. In this case, we will set up break points in a hybrid retrieval-augmented generation (RAG) pipeline. The pipeline combines BM25 and embedding-based retrieval methods, then uses a transformer-based reranker and an LLM to generate answers.

## Install packages
"""
logger.info("# Hybrid RAG Pipeline with Breakpoints")

# %%bash

# pip install haystack-ai>=2.16.0
# pip install "transformers[torch,sentencepiece]"
# pip install "sentence-transformers>=3.0.0"

"""
## Setup Ollama API keys
"""
logger.info("## Setup Ollama API keys")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter Ollama API key:")

"""
## Import Required Libraries

First, let's import all the necessary components from Haystack.
"""
logger.info("## Import Required Libraries")


"""
## Document Store Initializations

Let's create a simple document store with some sample documents and their embeddings.
"""
logger.info("## Document Store Initializations")

def indexing():
    """
    Indexing documents in a DocumentStore.
    """

    logger.debug("Indexing documents...")

    documents = [
        Document(content="My name is Jean and I live in Paris. The weather today is 25°C."),
        Document(content="My name is Mark and I live in Berlin. The weather today is 15°C."),
        Document(content="My name is Giorgio and I live in Rome. The weather today is 30°C."),
    ]

    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2", progress_bar=False)

    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")

    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})

    return document_store

"""
## A Hybrid Retrieval Pipeline

Now let's build a hybrid RAG pipeline.
"""
logger.info("## A Hybrid Retrieval Pipeline")

def hybrid_retrieval(doc_store):
    """
    A simple pipeline for hybrid retrieval using BM25 and embeddings.
    """

    query_embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", progress_bar=False)

    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question based on the given context information only. If the context is empty or just a '\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]


    rag_pipeline = Pipeline()

    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=doc_store), name="bm25_retriever")
    rag_pipeline.add_component(instance=query_embedder, name="query_embedder")
    rag_pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=doc_store), name="embedding_retriever")
    rag_pipeline.add_component(instance=DocumentJoiner(sort_by_score=False), name="doc_joiner")
    rag_pipeline.add_component(instance=TransformersSimilarityRanker(model="intfloat/simlm-msmarco-reranker", top_k=5), name="ranker")
    rag_pipeline.add_component(instance=ChatPromptBuilder(template=template, required_variables=["question", "documents"]), name="prompt_builder", )
    rag_pipeline.add_component(instance=OllamaChatGenerator(model="qwen3:4b-q4_K_M"), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

    rag_pipeline.connect("query_embedder", "embedding_retriever.query_embedding")
    rag_pipeline.connect("embedding_retriever", "doc_joiner.documents")
    rag_pipeline.connect("bm25_retriever", "doc_joiner.documents")
    rag_pipeline.connect("doc_joiner", "ranker.documents")
    rag_pipeline.connect("ranker", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("doc_joiner", "answer_builder.documents")

    return rag_pipeline

"""
## Running the pipeline with breakpoints

Now we demonstrate how to set breakpoints in a Haystack pipeline to inspect and debug the pipeline execution at specific points. Breakpoints allow you to pause execution, save the current state of pipeline, and later resume from where you left off.

We'll run the pipeline with a breakpoint set at the `query_embedder` component. This will save the pipeline state before executing the `query_embedder` and raise `PipelineBreakpointException` to stop execution.
"""
# Update the query_embedder breakpoint section
logger.info("## Running the pipeline with breakpoints")
break_point = Breakpoint(component_name="query_embedder", visit_count=0, snapshot_file_path=f"{OUTPUT_DIR}/snapshots/")
doc_store = indexing()
pipeline = hybrid_retrieval(doc_store)
question = "Where does Mark live?"
data = {
    "query_embedder": {"text": question},
    "bm25_retriever": {"query": question},
    "ranker": {"query": question, "top_k": 10},
    "prompt_builder": {"question": question},
    "answer_builder": {"query": question},
}

try:
    pipeline.run(data, break_point=break_point)
except BreakpointException as e:
    logger.info(f"Breakpoint triggered: {e}")
    # Find the latest query_embedder snapshot file
    snapshot_files = glob.glob(f"{OUTPUT_DIR}/snapshots/query_embedder_*.json")
    if not snapshot_files:
        logger.error("No query_embedder snapshot files found in snapshots directory")
        raise FileNotFoundError("No query_embedder snapshot files found")
    latest_snapshot = max(snapshot_files, key=os.path.getctime)
    logger.info(f"Loading query_embedder snapshot from: {latest_snapshot}")
    snapshot = load_pipeline_snapshot(latest_snapshot)
    result = pipeline.run(data={}, pipeline_snapshot=snapshot)
    logger.debug(result['answer_builder']['answers'][0].data)
    logger.debug(result['answer_builder']['answers'][0].meta)

"""
This run should be interruped with a `BreakpointException: Breaking at component query_embedder visit count 0` - and this will generate a JSON file in the "snapshots" directory containing a snapshot of the  before running the component `query_embedder`.

The snapshot files, named after the component associated with the breakpoint, can be inspected and edited, and later injected into a pipeline and resume the execution from the point where the breakpoint was triggered.
"""
logger.info("This run should be interruped with a `BreakpointException: Breaking at component query_embedder visit count 0` - and this will generate a JSON file in the \"snapshots\" directory containing a snapshot of the  before running the component `query_embedder`.")

# !ls snapshots/

"""
## Resuming from a break point

We can then resume a pipeline from its saved `pipeline_snapshot` by passing it to the Pipeline.run() method. This will run the pipeline to the end.
"""
logger.info("## Resuming from a break point")


snapshot = load_pipeline_snapshot(f"{OUTPUT_DIR}/snapshots/query_embedder_2025_07_26_12_58_26.json")
result = pipeline.run(data={}, pipeline_snapshot=snapshot)

logger.debug(result['answer_builder']['answers'][0].data)
logger.debug(result['answer_builder']['answers'][0].meta)

"""
## Advanced Use Cases for Pipeline Breakpoints

Here are some advanced scenarios where pipeline breakpoints can be particularly valuable:
1. Set a breakpoint at the LLM to try results of different prompts and iterate in real time.

2. Place a breakpoint after the document retriever to examine and modify retrieved documents.

3. Set a breakpoint before a component to inject gold-standard inputs and isolate whether issues stem from input quality or downstream logic.

To demonstrate the use case stated in point 1, we reuse the same query pipeline with a new question. First, we run the pipeline with the prompt that we originally passed to the prompt_builder. Then, we define a breakpoint at the prompt_builder to try an alternative prompt. This allows us to compare the results generated by different prompts without running the whole pipeline again.
"""
# Update the prompt_builder breakpoint section
logger.info("## Advanced Use Cases for Pipeline Breakpoints")
doc_store = indexing()
pipeline = hybrid_retrieval(doc_store)
question = "What's the temperature difference between the warmest and coldest city?"
data = {
    "query_embedder": {"text": question},
    "bm25_retriever": {"query": question},
    "ranker": {"query": question, "top_k": 10},
    "prompt_builder": {"question": question},
    "answer_builder": {"query": question},
}
break_point = Breakpoint(component_name="prompt_builder", visit_count=0, snapshot_file_path=f"{OUTPUT_DIR}/snapshots/")

try:
    pipeline.run(data, break_point=break_point)
except BreakpointException as e:
    logger.info(f"Breakpoint triggered: {e}")
    # Find the latest prompt_builder snapshot file
    snapshot_files = glob.glob(f"{OUTPUT_DIR}/snapshots/prompt_builder_*.json")
    if not snapshot_files:
        logger.error("No prompt_builder snapshot files found in snapshots directory")
        raise FileNotFoundError("No prompt_builder snapshot files found")
    latest_snapshot = max(snapshot_files, key=os.path.getctime)
    logger.info(f"Loading prompt_builder snapshot from: {latest_snapshot}")
    # Update the prompt_builder template in the snapshot (if needed)
    snapshot = load_pipeline_snapshot(latest_snapshot)
    result = pipeline.run(data={}, pipeline_snapshot=snapshot)
    logger.debug(result['answer_builder']['answers'][0].data)
    save_file(result, f"{OUTPUT_DIR}/result.json")
    save_file(result['answer_builder']['answers'][0].data, f"{OUTPUT_DIR}/results/answers.json")

"""
Now we can manually insert a different template into the `prompt_builder` and inspect the results. To do this, we update the template input within the `prompt_builder` component in the state file.
"""
logger.info("Now we can manually insert a different template into the `prompt_builder` and inspect the results. To do this, we update the template input within the `prompt_builder` component in the state file.")

template = ChatMessage.from_system(
    """You are a mathematical analysis assistant. Follow these steps:
    1. Identify all temperatures mentioned
    2. Find the maximum and minimum values
    3. Calculate their difference
    4. Format response as: 'The temperature difference is X°C (max Y°C in [city] - min Z°C in [city])'
    Use ONLY the information provided in the context."""
)

"""
Now we just load the snapshot file and resume the pipeline with the updated snapshot.
"""
logger.info("Now we just load the snapshot file and resume the pipeline with the updated snapshot.")

# !ls snapshots/prompt_builder*

# Replace the snapshot loading and pipeline resumption section
logger.info("\n\n## Resuming from a break point")
snapshot_files = glob.glob(f"{OUTPUT_DIR}/snapshots/prompt_builder_*.json")
if not snapshot_files:
    logger.error("No snapshot files found in snapshots directory")
    raise FileNotFoundError("No snapshot files found")
latest_snapshot = max(snapshot_files, key=os.path.getctime)
logger.info(f"Loading snapshot from: {latest_snapshot}")
snapshot = load_pipeline_snapshot(latest_snapshot)
result = pipeline.run(data={}, pipeline_snapshot=snapshot)
logger.debug(result['answer_builder']['answers'][0].data)
save_file(result, f"{OUTPUT_DIR}/result.json")
save_file(result['answer_builder']['answers'][0].data, f"{OUTPUT_DIR}/results/answers.json")

logger.info("\n\n[DONE]", bright=True)
