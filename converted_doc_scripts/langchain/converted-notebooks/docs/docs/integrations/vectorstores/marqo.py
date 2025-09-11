from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Marqo
from langchain_text_splitters import CharacterTextSplitter
import marqo
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
# Marqo

This notebook shows how to use functionality related to the Marqo vectorstore.

>[Marqo](https://www.marqo.ai/) is an open-source vector search engine. Marqo allows you to store and query multi-modal data such as text and images. Marqo creates the vectors for you using a huge selection of open-source models, you can also provide your own fine-tuned models and Marqo will handle the loading and inference for you.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

To run this notebook with our docker image please run the following commands first to get Marqo:

```
docker pull marqoai/marqo:latest
docker rm -f marqo
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```
"""
logger.info("# Marqo")

# %pip install --upgrade --quiet  marqo


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# if using marqo cloud replace with your endpoint (console.marqo.ai)
marqo_url = "http://localhost:8882"
marqo_  # if using marqo cloud replace with your api key (console.marqo.ai)

client = marqo.Client(url=marqo_url, api_key=marqo_api_key)

index_name = "langchain-demo"

docsearch = Marqo.from_documents(docs, index_name=index_name)

query = "What did the president say about Ketanji Brown Jackson"
result_docs = docsearch.similarity_search(query)

logger.debug(result_docs[0].page_content)

result_docs = docsearch.similarity_search_with_score(query)
logger.debug(result_docs[0][0].page_content, result_docs[0][1], sep="\n")

"""
## Additional features

One of the powerful features of Marqo as a vectorstore is that you can use indexes created externally. For example:

+ If you had a database of image and text pairs from another application, you can simply just use it in langchain with the Marqo vectorstore. Note that bringing your own multimodal indexes will disable the `add_texts` method.

+ If you had a database of text documents, you can bring it into the langchain framework and add more texts through `add_texts`.

The documents that are returned are customised by passing your own function to the `page_content_builder` callback in the search methods.

#### Multimodal Example
"""
logger.info("## Additional features")

index_name = "langchain-multimodal-demo"

try:
    client.delete_index(index_name)
except Exception:
    logger.debug(f"Creating {index_name}")

settings = {"treat_urls_and_pointers_as_images": True, "model": "ViT-L/14"}
client.create_index(index_name, **settings)
client.index(index_name).add_documents(
    [
        {
            "caption": "Bus",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
        },
        {
            "caption": "Plane",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
        },
    ],
)


def get_content(res):
    """Helper to format Marqo's documents into text to be used as page_content"""
    return f"{res['caption']}: {res['image']}"


docsearch = Marqo(client, index_name, page_content_builder=get_content)


query = "vehicles that fly"
doc_results = docsearch.similarity_search(query)

for doc in doc_results:
    logger.debug(doc.page_content)

"""
#### Text only example
"""
logger.info("#### Text only example")

index_name = "langchain-byo-index-demo"

try:
    client.delete_index(index_name)
except Exception:
    logger.debug(f"Creating {index_name}")

client.create_index(index_name)
client.index(index_name).add_documents(
    [
        {
            "Title": "Smartphone",
            "Description": "A smartphone is a portable computer device that combines mobile telephone "
            "functions and computing functions into one unit.",
        },
        {
            "Title": "Telephone",
            "Description": "A telephone is a telecommunications device that permits two or more users to"
            "conduct a conversation when they are too far apart to be easily heard directly.",
        },
    ],
)


def get_content(res):
    """Helper to format Marqo's documents into text to be used as page_content"""
    if "text" in res:
        return res["text"]
    return res["Description"]


docsearch = Marqo(client, index_name, page_content_builder=get_content)

docsearch.add_texts(["This is a document that is about elephants"])

query = "modern communications devices"
doc_results = docsearch.similarity_search(query)

logger.debug(doc_results[0].page_content)

query = "elephants"
doc_results = docsearch.similarity_search(
    query, page_content_builder=get_content)

logger.debug(doc_results[0].page_content)

"""
## Weighted Queries

We also expose marqos weighted queries which are a powerful way to compose complex semantic searches.
"""
logger.info("## Weighted Queries")

query = {"communications devices": 1.0}
doc_results = docsearch.similarity_search(query)
logger.debug(doc_results[0].page_content)

query = {"communications devices": 1.0, "technology post 2000": -1.0}
doc_results = docsearch.similarity_search(query)
logger.debug(doc_results[0].page_content)

"""
# Question Answering with Sources

This section shows how to use Marqo as part of a `RetrievalQAWithSourcesChain`. Marqo will perform the searches for information in the sources.
"""
logger.info("# Question Answering with Sources")

# import getpass


# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

with open("../../how_to/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

index_name = "langchain-qa-with-retrieval"
docsearch = Marqo.from_documents(docs, index_name=index_name)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    Ollama(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
)

chain(
    {"question": "What did the president say about Justice Breyer"},
    return_only_outputs=True,
)

logger.info("\n\n[DONE]", bright=True)
