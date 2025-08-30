from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder
from haystack_integrations.components.embedders.nvidia import NvidiaTextEmbedder
from haystack_integrations.components.generators.nvidia import NvidiaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from jet.logger import CustomLogger
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
# Haystack RAG Pipeline with Self-Deployed AI models using NVIDIA NIMs

In this notebook, we will build a Haystack Retrieval Augmented Generation (RAG) Pipeline using self-hosted AI models with NVIDIA Inference Microservices or NIMs.

The notebook is associated with a technical blog demonstrating the steps to deploy NVIDIA NIMs with Haystack into production.

The code examples expect the LLM Generator and Retrieval Embedding AI models already deployed using NIMs microservices following the procedure described in the technical blog.

You can also substitute the calls to NVIDIA NIMs with the same AI models hosted by NVIDIA on [ai.nvidia.com](https://ai.nvidia.com).
"""
logger.info("# Haystack RAG Pipeline with Self-Deployed AI models using NVIDIA NIMs")

# !pip install haystack-ai
# !pip install nvidia-haystack
# !pip install --upgrade setuptools==67.0
# !pip install pip install pydantic==1.9.0
# !pip install pypdf
# !pip install hayhooks
# !pip install qdrant-haystack

"""
For the Haystack RAG pipeline, we will use the Qdrant Vector Database and the self-hosted [*meta-llama3-8b-instruct*](https://build.nvidia.com/explore/discover#llama3-70b) for the LLM Generator and [*NV-Embed-QA*](https://build.nvidia.com/nvidia/embed-qa-4) for the embedder.

In the next cell, We will set the domain names and URLs of the self-deployed NVIDIA NIMs as well as the QdrantDocumentStore URL. Adjust these according to your setup.
"""
logger.info("For the Haystack RAG pipeline, we will use the Qdrant Vector Database and the self-hosted [*meta-llama3-8b-instruct*](https://build.nvidia.com/explore/discover#llama3-70b) for the LLM Generator and [*NV-Embed-QA*](https://build.nvidia.com/nvidia/embed-qa-4) for the embedder.")


llm_nim_model_name = "meta-llama3-8b-instruct"
llm_nim_base_url = "http://nims.example.com/llm"

embedding_nim_model = "NV-Embed-QA"
embedding_nim_api_url = "http://nims.example.com/embedding"

qdrant_endpoint = "http://vectordb.example.com:30030"

"""
## 1. Check Deployments

Let's first check the Vector database and the self-deployed models with NVIDIA NIMs in our environment. Have a look at the technical blog for the steps NIM deployment.

We can check the AI models deployed with NIMs and the Qdrant database using simple *curl* commands.

### 1.1 Check the LLM Generator NIM
"""
logger.info("## 1. Check Deployments")

# ! curl '{llm_nim_base_url}/v1/models' -H 'Accept: application/json'

"""
### 1.2 Check the Retreival Embedding NIM
"""
logger.info("### 1.2 Check the Retreival Embedding NIM")

# ! curl '{embedding_nim_api_url}/v1/models' -H 'Accept: application/json'

"""
### 1.3 Check the Qdrant Database
"""
logger.info("### 1.3 Check the Qdrant Database")

# ! curl '{qdrant_endpoint}' -H 'Accept: application/json'

os.environ["NVIDIA_API_KEY"] = ""

"""
## 2. Perform Indexing

Let's first initialize the Qdrant vector database, create the Haystack indexing pipeline and upload pdf examples. We will use the self-deployed embedder AI model with NIM.
"""
logger.info("## 2. Perform Indexing")


document_store = QdrantDocumentStore(embedding_dim=1024, url=qdrant_endpoint)

embedder = NvidiaDocumentEmbedder(
    model=embedding_nim_model,
    api_url=f"{embedding_nim_api_url}/v1"
)

converter = PyPDFToDocument()
cleaner = DocumentCleaner()
splitter = DocumentSplitter(split_by='word', split_length=100)
writer = DocumentWriter(document_store)

indexing = Pipeline()
indexing.add_component("converter", converter)
indexing.add_component("cleaner", cleaner)
indexing.add_component("splitter", splitter)
indexing.add_component("embedder", embedder)
indexing.add_component("writer", writer)

indexing.connect("converter", "cleaner")
indexing.connect("cleaner", "splitter")
indexing.connect("splitter", "embedder")
indexing.connect("embedder", "writer")

"""
We will upload in the vector database a PDF research paper about **ChipNeMo** from NVIDIA, a domain specific LLM for Chip design.
The paper is available [here](https://raw.githubusercontent.com/deepset-ai/haystack-cookbook/main/data/rag-with-nims/ChipNeMo.pdf).
"""
logger.info("We will upload in the vector database a PDF research paper about **ChipNeMo** from NVIDIA, a domain specific LLM for Chip design.")

document_sources = ["./data/ChipNeMo.pdf"]

indexing.run({"converter": {"sources": document_sources}})

"""
It is possible to check the Qdrant database deployments through the Web UI. We can check the embeddings stored on the dashboard available [qdrant_endpoint/dashboard](http://vectordb.example.com:30030/dashboard
)

![](/images/embeddings-1.png)

<center><img src="https://raw.githubusercontent.com/deepset-ai/haystack-cookbook/main/data/rag-with-nims/embeddings-1.png"></center>
<center><img src="https://raw.githubusercontent.com/deepset-ai/haystack-cookbook/main/data/rag-with-nims/embeddings-2.png"></center>

## 3. Create the RAG Pipeline

Let's now create the Haystack RAG pipeline. We will initialize the LLM generator with the self-deployed LLM with NIM.
"""
logger.info("## 3. Create the RAG Pipeline")


embedder = NvidiaTextEmbedder(
    model=embedding_nim_model,
    api_url=f"{embedding_nim_api_url}/v1"
)

generator = NvidiaGenerator(
    model=llm_nim_model_name,
    api_url=f"{llm_nim_base_url}/v1",
    model_arguments={
        "temperature": 0.5,
        "top_p": 0.7,
        "max_tokens": 2048,
    },
)

retriever = QdrantEmbeddingRetriever(document_store=document_store)

prompt = """Answer the question given the context.
Question: {{ query }}
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
Answer:"""
prompt_builder = PromptBuilder(template=prompt)

rag = Pipeline()
rag.add_component("embedder", embedder)
rag.add_component("retriever", retriever)
rag.add_component("prompt", prompt_builder)
rag.add_component("generator", generator)

rag.connect("embedder.embedding", "retriever.query_embedding")
rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt", "generator")

"""
Let's now request the RAG pipeline asking a question about the ChipNemo model.
"""
logger.info("Let's now request the RAG pipeline asking a question about the ChipNemo model.")

question = "Describe chipnemo in detail?"
result = rag.run(
    {
        "embedder": {"text": question},
        "prompt": {"query": question},
    }, include_outputs_from=["prompt"]
)
logger.debug(result["generator"]["replies"][0])

"""
This notebook shows how to build a Haystack RAG pipeline using self-deployed generative AI models with NVIDIA Inference Microservices (NIMs).

Please check the documentation on how to deploy NVIDIA NIMs in your own environment.

For experimentation purpose, it is also possible to substitute the self-deployed models with NIMs hosted by NVIDIA at [ai.nvidia.com](https://ai.nvidia.com).
"""
logger.info("This notebook shows how to build a Haystack RAG pipeline using self-deployed generative AI models with NVIDIA Inference Microservices (NIMs).")

logger.info("\n\n[DONE]", bright=True)