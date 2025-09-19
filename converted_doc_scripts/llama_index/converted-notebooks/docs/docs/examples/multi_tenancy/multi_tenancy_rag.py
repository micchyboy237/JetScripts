from IPython.display import HTML
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_tenancy/multi_tenancy_rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Tenancy RAG with LlamaIndex

In this notebook you will look into building Multi-Tenancy RAG System using LlamaIndex.

1. Setup
2. Download Data
3. Load Data
4. Create Index
5. Create Ingestion Pipeline
6. Update Metadata and Insert documents
7. Define Query Engines for each user
8. Querying

## Setup

 You should ensure you have `llama-index` and `pypdf` is installed.
"""
logger.info("# Multi-Tenancy RAG with LlamaIndex")

# !pip install llama-index pypdf

"""
### Set OllamaFunctionCalling Key
"""
logger.info("### Set OllamaFunctionCalling Key")


# os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"



"""
## Download Data

We will use `An LLM Compiler for Parallel Function Calling` and `Dense X Retrieval: What Retrieval Granularity Should We Use?` papers for the demonstartions.
"""
logger.info("## Download Data")

# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2312.04511.pdf" -O "llm_compiler.pdf"
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2312.06648.pdf" -O "dense_x_retrieval.pdf"

"""
## Load Data
"""
logger.info("## Load Data")

reader = SimpleDirectoryReader(input_files=["dense_x_retrieval.pdf"])
documents_jerry = reader.load_data()

reader = SimpleDirectoryReader(input_files=["llm_compiler.pdf"])
documents_ravi = reader.load_data()

"""
## Create an Empty Index
"""
logger.info("## Create an Empty Index")

index = VectorStoreIndex.from_documents(documents=[])

"""
## Create Ingestion Pipeline
"""
logger.info("## Create Ingestion Pipeline")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=20),
    ]
)

"""
## Update Metadata and Insert Documents
"""
logger.info("## Update Metadata and Insert Documents")

for document in documents_jerry:
    document.metadata["user"] = "Jerry"

nodes = pipeline.run(documents=documents_jerry)
index.insert_nodes(nodes)

for document in documents_ravi:
    document.metadata["user"] = "Ravi"

nodes = pipeline.run(documents=documents_ravi)
index.insert_nodes(nodes)

"""
## Define Query Engines

Define query engines for both the users with necessary filters.
"""
logger.info("## Define Query Engines")

jerry_query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="user",
                value="Jerry",
            )
        ]
    ),
    similarity_top_k=3,
)

ravi_query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="user",
                value="Ravi",
            )
        ]
    ),
    similarity_top_k=3,
)

"""
## Querying
"""
logger.info("## Querying")

response = jerry_query_engine.query(
    "what are propositions mentioned in the paper?"
)
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

response = ravi_query_engine.query("what are steps involved in LLMCompiler?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

response = jerry_query_engine.query("what are steps involved in LLMCompiler?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

logger.info("\n\n[DONE]", bright=True)