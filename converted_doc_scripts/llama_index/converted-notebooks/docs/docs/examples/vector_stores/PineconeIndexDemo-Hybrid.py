from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.sparse_embeddings.fastembed import FastEmbedSparseEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pinecone Vector Store - Hybrid Search

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Pinecone Vector Store - Hybrid Search")

# %pip install llama-index-vector-stores-pinecone "transformers[torch]"

"""
#### Creating a Pinecone Index
"""
logger.info("#### Creating a Pinecone Index")



os.environ["PINECONE_API_KEY"] = "..."
# os.environ["OPENAI_API_KEY"] = "sk-..."

api_key = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)

pc.delete_index("quickstart")

pc.create_index(
    name="quickstart",
    dimension=1536,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

pinecone_index = pc.Index("quickstart")

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load documents, build the PineconeVectorStore

When `add_sparse_vector=True`, the `PineconeVectorStore` will compute sparse vectors for each document.

By default, it is using simple token frequency for the sparse vectors. But, you can also specify a custom sparse embedding model.
"""
logger.info("#### Load documents, build the PineconeVectorStore")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


# if "OPENAI_API_KEY" not in os.environ:
#     raise EnvironmentError(f"Environment variable OPENAI_API_KEY is not set")

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    add_sparse_vector=True,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
#### Query Index

May need to wait a minute or two for the index to be ready
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine(vector_store_query_mode="hybrid")
response = query_engine.query("What happened at Viaweb?")

display(Markdown(f"<b>{response}</b>"))

"""
## Changing the sparse embedding model
"""
logger.info("## Changing the sparse embedding model")

# %pip install llama-index-sparse-embeddings-fastembed

vector_store.clear()


sparse_embedding_model = FastEmbedSparseEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    add_sparse_vector=True,
    sparse_embedding_model=sparse_embedding_model,
)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
Wait a mininute for things to upload..
"""
logger.info("Wait a mininute for things to upload..")

response = query_engine.query("What happened at Viaweb?")
display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)