from jet.logger import logger
from langchain_community.vectorstores import LLMRails
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
# LLMRails

>[LLMRails](https://www.llmrails.com/) is a API platform for building GenAI applications. It provides an easy-to-use API for document indexing and querying that is managed by LLMRails and is optimized for performance and accuracy. 
See the [LLMRails API documentation ](https://docs.llmrails.com/) for more information on how to use the API.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the `LLMRails`'s integration with langchain.
Note that unlike many other integrations in this category, LLMRails provides an end-to-end managed service for retrieval augmented generation, which includes:
1. A way to extract text from document files and chunk them into sentences.
2. Its own embeddings model and vector store - each text segment is encoded into a vector embedding and stored in the LLMRails internal vector store
3. A query service that automatically encodes the query into embedding, and retrieves the most relevant text segments (including support for [Hybrid Search](https://docs.llmrails.com/datastores/search))

All of these are supported in this LangChain integration.

# Setup

You will need a LLMRails account to use LLMRails with LangChain. To get started, use the following steps:
1. [Sign up](https://console.llmrails.com/signup) for a LLMRails account if you don't already have one.
2. Next you'll need to create API keys to access the API. Click on the **"API Keys"** tab in the corpus view and then the **"Create API Key"** button. Give your key a name. Click "Create key" and you now have an active API key. Keep this key confidential. 

To use LangChain with LLMRails, you'll need to have this value: api_key.
You can provide those to LangChain in two ways:

1. Include in your environment these two variables: `LLM_RAILS_API_KEY`, `LLM_RAILS_DATASTORE_ID`.

# > For example, you can set these variables using os.environ and getpass as follows:

```python
# import getpass

# os.environ["LLM_RAILS_API_KEY"] = getpass.getpass("LLMRails API Key:")
# os.environ["LLM_RAILS_DATASTORE_ID"] = getpass.getpass("LLMRails Datastore Id:")
```

1. Provide them as arguments when creating the LLMRails vectorstore object:

```python
vectorstore = LLMRails(
    api_key=llm_rails_api_key,
    datastore_id=datastore_id
)
```

## Adding text

For adding text to your datastore first you have to go to [Datastores](https://console.llmrails.com/datastores) page and create one. Click Create Datastore button and choose a name and embedding model for your datastore. Then get your datastore id from newly created  datastore settings.
"""
logger.info("# LLMRails")

# %pip install tika



os.environ["LLM_RAILS_DATASTORE_ID"] = "Your datastore id "
os.environ["LLM_RAILS_API_KEY"] = "Your API Key"

llm_rails = LLMRails.from_texts(["Your text here"])

"""
## Similarity search

The simplest scenario for using LLMRails is to perform a similarity search.
"""
logger.info("## Similarity search")

query = "What do you plan to do about national security?"
found_docs = llm_rails.similarity_search(query, k=5)

logger.debug(found_docs[0].page_content)

"""
## Similarity search with score

Sometimes we might want to perform the search, but also obtain a relevancy score to know how good is a particular result.
"""
logger.info("## Similarity search with score")

query = "What is your approach to national defense"
found_docs = llm_rails.similarity_search_with_score(
    query,
    k=5,
)

document, score = found_docs[0]
logger.debug(document.page_content)
logger.debug(f"\nScore: {score}")

"""
## LLMRails as a Retriever

LLMRails, as all the other LangChain vectorstores, is most often used as a LangChain Retriever:
"""
logger.info("## LLMRails as a Retriever")

retriever = llm_rails.as_retriever()
retriever

query = "What is your approach to national defense"
retriever.invoke(query)

logger.info("\n\n[DONE]", bright=True)