from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.premai import PremAIEmbeddings
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
<a href="https://colab.research.google.com/drive/176IfpC2akqWOhDpVSnAA_eLEbzt4a5Fw?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# PremAI Embeddings

[PremAI](https://premai.io/) is an all-in-one platform that simplifies the creation of robust, production-ready applications powered by Generative AI. By streamlining the development process, PremAI allows you to concentrate on enhancing user experience and driving overall growth for your application. You can quickly start using our platform [here](https://docs.premai.io/quick-start).

In this section we are going to dicuss how we can get access to different embedding model using `PremEmbeddings` with llama-index

## Installation and setup

We start by installing `llama-index` and `premai-sdk`. You can type the following command to install:

```bash
pip install premai llama-index
```

Before proceeding further, please make sure that you have made an account on PremAI and already created a project. If not, please refer to the [quick start](https://docs.premai.io/introduction) guide to get started with the PremAI platform. Create your first project and grab your API key.
"""
logger.info("# PremAI Embeddings")

# %pip install llama-index-llms-premai


"""
## Setup PremAIEmbeddings instance in LlamaIndex 

Once we imported our required modules, let's setup our client. For now let's assume that our `project_id` is `8`. But make sure you use your project-id, otherwise it will throw error.

In order to use llama-index with PremAI, you do not need to pass any model name or set any parameters with our chat-client. By default it will use the model name and parameters used in the [LaunchPad](https://docs.premai.io/get-started/launchpad).

We support lots of state of the art embedding models. You can view our list of supported LLMs and embedding models [here](https://docs.premai.io/get-started/supported-models). For now let's go for `text-embedding-3-large` model for this example.
"""
logger.info("## Setup PremAIEmbeddings instance in LlamaIndex")

# import getpass

if os.environ.get("PREMAI_API_KEY") is None:
#     os.environ["PREMAI_API_KEY"] = getpass.getpass("PremAI API Key:")

prem_embedding = PremAIEmbeddings(
    project_id=8, model_name="text-embedding-3-large"
)

"""
## Calling the Embedding Model

Now you are all set. Now let's start using our embedding model with a single query followed by multiple queries (which is also called as a document)
"""
logger.info("## Calling the Embedding Model")

query = "Hello, this is a test query"
query_result = prem_embedding.get_text_embedding(query)

logger.debug(f"Dimension of embeddings: {len(query_result)}")

query_result[:5]

logger.info("\n\n[DONE]", bright=True)