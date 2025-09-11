from jet.logger import logger
from langchain_community.embeddings import PremAIEmbeddings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# PremAI

[PremAI](https://premai.io/) is an all-in-one platform that simplifies the creation of robust, production-ready applications powered by Generative AI. By streamlining the development process, PremAI allows you to concentrate on enhancing user experience and driving overall growth for your application. You can quickly start using our platform [here](https://docs.premai.io/quick-start).

### Installation and setup

We start by installing `langchain` and `premai-sdk`. You can type the following command to install:

```bash
pip install premai langchain
```

Before proceeding further, please make sure that you have made an account on PremAI and already created a project. If not, please refer to the [quick start](https://docs.premai.io/introduction) guide to get started with the PremAI platform. Create your first project and grab your API key.

## PremEmbeddings

In this section we are going to dicuss how we can get access to different embedding model using `PremEmbeddings` with LangChain. Let's start by importing our modules and setting our API Key.
"""
logger.info("# PremAI")


"""
Once we imported our required modules, let's setup our client. For now let's assume that our `project_id` is `8`. But make sure you use your project-id, otherwise it will throw error.

> Note: Setting `model_name` argument in mandatory for PremAIEmbeddings unlike [ChatPremAI.](https://python.langchain.com/v0.1/docs/integrations/chat/premai/)
"""
logger.info("Once we imported our required modules, let's setup our client. For now let's assume that our `project_id` is `8`. But make sure you use your project-id, otherwise it will throw error.")

# import getpass

if os.environ.get("PREMAI_API_KEY") is None:
#     os.environ["PREMAI_API_KEY"] = getpass.getpass("PremAI API Key:")

model = "text-embedding-3-large"
embedder = PremAIEmbeddings(project_id=8, model=model)

"""
We support lots of state of the art embedding models. You can view our list of supported LLMs and embedding models [here](https://docs.premai.io/get-started/supported-models). For now let's go for `text-embedding-3-large` model for this example.
"""
logger.info("We support lots of state of the art embedding models. You can view our list of supported LLMs and embedding models [here](https://docs.premai.io/get-started/supported-models). For now let's go for `text-embedding-3-large` model for this example.")

query = "Hello, this is a test query"
query_result = embedder.embed_query(query)


logger.debug(query_result[:5])

"""
Finally let's embed a document
"""
logger.info("Finally let's embed a document")

documents = ["This is document1", "This is document2", "This is document3"]

doc_result = embedder.embed_documents(documents)


logger.debug(doc_result[0][:5])

logger.info("\n\n[DONE]", bright=True)