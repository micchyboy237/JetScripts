from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.extractors import DocumentContextExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore.simple_docstore import (
SimpleDocumentStore,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
# Contextual Retrieval With Llama Index

This notebook covers contextual retrieval with llama_index DocumentContextExtractor

Based on an Anthropic [blost post](https://www.anthropic.com/news/contextual-retrieval), the concept is to:
1. Use an LLM to generate a 'context' for each chunk based on the entire document
2. embed the chunk + context together
3. reap the benefits of higher RAG accuracy

While you can also do this manually, the DocumentContextExtractor offers a lot of convenience and error handling, plus you can integrate it into your llama_index pipelines! Let's get started.

NOTE: This notebook costs about $0.02 everytime you run it.

# Install Packages
"""
logger.info("# Contextual Retrieval With Llama Index")

# %pip install llama-index
# %pip install llama-index-readers-file
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-ollama

"""
# Setup an LLM
You can use the MockLLM or you can use a real LLM of your choice here. flash 2 and qwen3-1.7b-4bit-mini work well.
"""
logger.info("# Setup an LLM")


# OPENAI_API_KEY = "sk-..."
# llm = MLX(model="qwen3-1.7b-4bit-mini", api_key=OPENAI_API_KEY)
Settings.llm = llm

"""
# Setup a data pipeline

 we'll need an embedding model, an index store, a vectore store, and a way to split tokens.

### Build Pipeline & Index
"""
logger.info("# Setup a data pipeline")


docstore = SimpleDocumentStore()
embed_model = HuggingFaceEmbedding(model_name="baai/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(docstore=docstore)
storage_context_no_extra_context = StorageContext.from_defaults()
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=256, chunk_overlap=10
)

"""
#### DocumentContextExtractor
"""
logger.info("#### DocumentContextExtractor")


context_extractor = DocumentContextExtractor(
    docstore=docstore,
    max_context_length=128000,
    llm=llm,  # default to Settings.llm
    oversized_document_strategy="warn",
    max_output_tokens=100,
    key="context",
    prompt=DocumentContextExtractor.SUCCINCT_CONTEXT_PROMPT,
)

"""
# Load Data
"""
logger.info("# Load Data")

# !wget "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay_ambiguated.txt" -O "paul_graham_essay_ambiguated.txt"


reader = SimpleDirectoryReader(
    input_files=["./paul_graham_essay_ambiguated.txt"]
)
documents = reader.load_data()

"""
# Run the pipeline, then search
"""
logger.info("# Run the pipeline, then search")

# import nest_asyncio

# nest_asyncio.apply()

storage_context.docstore.add_documents(documents)
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    embed_model=embed_model,
    transformations=[text_splitter, context_extractor],
)

index_nocontext = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context_no_extra_context,
    embed_model=embed_model,
    transformations=[text_splitter],
)

test_question = "Which chunks of text discuss the IBM 704?"
retriever = index.as_retriever(similarity_top_k=2)
nodes_fromcontext = retriever.retrieve(test_question)

retriever_nocontext = index_nocontext.as_retriever(similarity_top_k=2)
nodes_nocontext = retriever_nocontext.retrieve(test_question)

logger.debug("==========")
logger.debug("NO CONTEXT")
for i, node in enumerate(nodes_nocontext, 1):
    logger.debug(f"\nChunk {i}:")
    logger.debug(f"Score: {node.score}")  # Similarity score
    logger.debug(f"Content: {node.node.text}")  # The actual text content

logger.debug("==========")
logger.debug("WITH CONTEXT")
for i, node in enumerate(nodes_fromcontext, 1):
    logger.debug(f"\nChunk {i}:")
    logger.debug(f"Score: {node.score}")  # Similarity score
    logger.debug(f"Content: {node.node.text}")  # The actual text content

logger.info("\n\n[DONE]", bright=True)