from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.predibase import PredibaseLLM
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/predibase.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Predibase

This notebook shows how you can use Predibase-hosted LLM's within Llamaindex. You can add [Predibase](https://predibase.com) to your existing Llamaindex worklow to: 
1. Deploy and query pre-trained or custom open source LLM’s without the hassle
2. Operationalize an end-to-end Retrieval Augmented Generation (RAG) system
3. Fine-tune your own LLM in just a few lines of code

## Getting Started
1. Sign up for a free Predibase account [here](https://predibase.com/free-trial)
2. Create an Account
3. Go to Settings > My profile and Generate a new API Token.
"""
logger.info("# Predibase")

# %pip install llama-index-llms-predibase

# !pip install llama-index --quiet
# !pip install predibase --quiet
# !pip install sentence-transformers --quiet


os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"

"""
## Flow 1: Query Predibase LLM directly
"""
logger.info("## Flow 1: Query Predibase LLM directly")

llm = PredibaseLLM(
    model_name="mistral-7b",
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="e2e_nlg",  # adapter_id is optional
    adapter_version=1,  # optional parameter (applies to Predibase only)
    api_token=None,  # optional parameter for accessing services hosting adapters (e.g., HuggingFace)
    max_new_tokens=512,
    temperature=0.3,
)

llm = PredibaseLLM(
    model_name="mistral-7b",
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="predibase/e2e_nlg",  # adapter_id is optional
    api_token=os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    ),  # optional parameter for accessing services hosting adapters (e.g., HuggingFace)
    max_new_tokens=512,
    temperature=0.3,
)

result = llm.complete("Can you recommend me a nice dry white wine?")
logger.debug(result)

"""
## Flow 2: Retrieval Augmented Generation (RAG) with Predibase LLM
"""
logger.info("## Flow 2: Retrieval Augmented Generation (RAG) with Predibase LLM")


"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Load Documents
"""
logger.info("### Load Documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
### Configure Predibase LLM
"""
logger.info("### Configure Predibase LLM")

llm = PredibaseLLM(
    model_name="mistral-7b",
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="e2e_nlg",  # adapter_id is optional
    api_token=None,  # optional parameter for accessing services hosting adapters (e.g., HuggingFace)
    temperature=0.3,
    context_window=1024,
)

llm = PredibaseLLM(
    model_name="mistral-7b",
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="predibase/e2e_nlg",  # adapter_id is optional
    api_token=os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    ),  # optional parameter for accessing services hosting adapters (e.g., HuggingFace)
    temperature=0.3,
    context_window=1024,
)

embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
splitter = SentenceSplitter(chunk_size=1024)

"""
### Setup and Query Index
"""
logger.info("### Setup and Query Index")

index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], embed_model=embed_model
)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What did the author do growing up?")

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)