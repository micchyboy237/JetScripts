from jet.logger import logger
from langchain_community.retrievers import KNNRetriever
from langchain_voyageai import VoyageAIEmbeddings
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
# Voyage AI

>[Voyage AI](https://www.voyageai.com/) provides cutting-edge embedding/vectorizations models.

Let's load the Voyage AI Embedding class. (Install the LangChain partner package with `pip install langchain-voyageai`)
"""
logger.info("# Voyage AI")


"""
Voyage AI utilizes API keys to monitor usage and manage permissions. To obtain your key, create an account on our [homepage](https://www.voyageai.com). Then, create a VoyageEmbeddings model with your API key. You can use any of the following models: ([source](https://docs.voyageai.com/docs/embeddings)):

- `voyage-3-large`
- `voyage-3`
- `voyage-3-lite`
- `voyage-large-2`
- `voyage-code-2`
- `voyage-2`
- `voyage-law-2`
- `voyage-large-2-instruct`
- `voyage-finance-2`
- `voyage-multilingual-2`
"""
logger.info("Voyage AI utilizes API keys to monitor usage and manage permissions. To obtain your key, create an account on our [homepage](https://www.voyageai.com). Then, create a VoyageEmbeddings model with your API key. You can use any of the following models: ([source](https://docs.voyageai.com/docs/embeddings)):")

embeddings = VoyageAIEmbeddings(
    voyage_model="voyage-law-2"
)

"""
Prepare the documents and use `embed_documents` to get their embeddings.
"""
logger.info("Prepare the documents and use `embed_documents` to get their embeddings.")

documents = [
    "Caching embeddings enables the storage or temporary caching of embeddings, eliminating the necessity to recompute them each time.",
    "An LLMChain is a chain that composes basic LLM functionality. It consists of a PromptTemplate and a language model (either an LLM or chat model). It formats the prompt template using the input key values provided (and also memory key values, if available), passes the formatted string to LLM and returns the LLM output.",
    "A Runnable represents a generic unit of work that can be invoked, batched, streamed, and/or transformed.",
]

documents_embds = embeddings.embed_documents(documents)

documents_embds[0][:5]

"""
Similarly, use `embed_query` to embed the query.
"""
logger.info("Similarly, use `embed_query` to embed the query.")

query = "What's an LLMChain?"

query_embd = embeddings.embed_query(query)

query_embd[:5]

"""
## A minimalist retrieval system

The main feature of the embeddings is that the cosine similarity between two embeddings captures the semantic relatedness of the corresponding original passages. This allows us to use the embeddings to do semantic retrieval / search.

We can find a few closest embeddings in the documents embeddings based on the cosine similarity, and retrieve the corresponding document using the `KNNRetriever` class from LangChain.
"""
logger.info("## A minimalist retrieval system")


retriever = KNNRetriever.from_texts(documents, embeddings)

result = retriever.invoke(query)
top1_retrieved_doc = result[0].page_content  # return the top1 retrieved result

logger.debug(top1_retrieved_doc)

logger.info("\n\n[DONE]", bright=True)