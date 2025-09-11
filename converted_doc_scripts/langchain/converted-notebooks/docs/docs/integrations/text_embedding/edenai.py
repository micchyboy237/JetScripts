from jet.logger import logger
from langchain_community.embeddings.edenai import EdenAiEmbeddings
import numpy as np
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
# EDEN AI

Eden AI is revolutionizing the AI landscape by uniting the best AI providers, empowering users to unlock limitless possibilities and tap into the true potential of artificial intelligence. With an all-in-one comprehensive and hassle-free platform, it allows users to deploy AI features to production lightning fast, enabling effortless access to the full breadth of AI capabilities via a single API. (website: https://edenai.co/)

This example goes over how to use LangChain to interact with Eden AI embedding models

-----------------------------------------------------------------------------------

Accessing the EDENAI's API requires an API key, 

which you can get by creating an account https://app.edenai.run/user/register  and heading here https://app.edenai.run/admin/account/settings

Once we have a key we'll want to set it as an environment variable by running:

```shell
export EDENAI_API_KEY="..."
```

If you'd prefer not to set an environment variable you can pass the key in directly via the edenai_api_key named parameter

 when initiating the EdenAI embedding class:
"""
logger.info("# EDEN AI")


embeddings = EdenAiEmbeddings(edenai_provider="...")

"""
## Calling a model

The EdenAI API brings together various providers.

To access a specific model, you can simply use the "provider" when calling.
"""
logger.info("## Calling a model")

embeddings = EdenAiEmbeddings(provider="ollama")

docs = ["It's raining right now", "cats are cute"]
document_result = embeddings.embed_documents(docs)

query = "my umbrella is broken"
query_result = embeddings.embed_query(query)


query_numpy = np.array(query_result)
for doc_res, doc in zip(document_result, docs):
    document_numpy = np.array(doc_res)
    similarity = np.dot(query_numpy, document_numpy) / (
        np.linalg.norm(query_numpy) * np.linalg.norm(document_numpy)
    )
    logger.debug(f'Cosine similarity between "{doc}" and query: {similarity}')

logger.info("\n\n[DONE]", bright=True)