from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms.llama_utils import (
messages_to_prompt,
completion_to_prompt,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/SimpleIndexDemoLlama2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Llama2 + VectorStoreIndex

This notebook walks through the proper setup to use llama-2 with LlamaIndex. Specifically, we look at using a vector store index.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Llama2 + VectorStoreIndex")

# %pip install llama-index-llms-replicate

# !pip install llama-index

"""
### Keys
"""
logger.info("### Keys")


# os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_TOKEN"

"""
### Load documents, build the VectorStoreIndex
"""
logger.info("### Load documents, build the VectorStoreIndex")




LLAMA_13B_V2_CHAT = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"


def custom_completion_to_prompt(completion: str) -> str:
    return completion_to_prompt(
        completion,
        system_prompt=(
            "You are a Q&A assistant. Your goal is to answer questions as "
            "accurately as possible is the instructions and context provided."
        ),
    )


llm = Replicate(
    model=LLAMA_13B_V2_CHAT,
    temperature=0.01,
    context_window=4096,
    completion_to_prompt=custom_completion_to_prompt,
    messages_to_prompt=messages_to_prompt,
)


Settings.llm = llm

"""
Download Data
"""
logger.info("Download Data")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents)

"""
## Querying
"""
logger.info("## Querying")

query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
### Streaming Support
"""
logger.info("### Streaming Support")

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("What happened at interleaf?")
for token in response.response_gen:
    logger.debug(token, end="")

logger.info("\n\n[DONE]", bright=True)