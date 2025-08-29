from jet.models.config import MODELS_CACHE_DIR
from IPython.display import Markdown, display
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.response.notebook_utils import (
display_source_node,
display_image_uris,
)
from llama_index.core.schema import ImageNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pathlib import Path
import chromadb
import openai
import os
import requests
import shutil
import urllib.request


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/ChromaMultiModalDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chroma Multi-Modal Demo with LlamaIndex

>[Chroma](https://docs.trychroma.com/getting-started) is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0.

<a href="https://discord.gg/MMeYNTmh3x" target="_blank">
      <img src="https://img.shields.io/discord/1073293645303795742" alt="Discord">
  </a>&nbsp;&nbsp;
  <a href="https://github.com/chroma-core/chroma/blob/master/LICENSE" target="_blank">
      <img src="https://img.shields.io/static/v1?label=license&message=Apache 2.0&color=white" alt="License">
  </a>&nbsp;&nbsp;
  <img src="https://github.com/chroma-core/chroma/actions/workflows/chroma-integration-test.yml/badge.svg?branch=main" alt="Integration Tests">

- [Website](https://www.trychroma.com/)
- [Documentation](https://docs.trychroma.com/)
- [Twitter](https://twitter.com/trychroma)
- [Discord](https://discord.gg/MMeYNTmh3x)

Chroma is fully-typed, fully-tested and fully-documented.

Install Chroma with:

```sh
pip install chromadb
```

Chroma runs in various modes. See below for examples of each integrated with LangChain.
- `in-memory` - in a python script or jupyter notebook
- `in-memory with persistence` - in a script or notebook and save/load to disk
- `in a docker container` - as a server running your local machine or in the cloud

Like any other database, you can: 
- `.add` 
- `.get` 
- `.update`
- `.upsert`
- `.delete`
- `.peek`
- and `.query` runs the similarity search.

View full docs at [docs](https://docs.trychroma.com/reference/Collection).

## Basic Example

In this basic example, we take the a Paul Graham essay, split it into chunks, embed it using an open-source embedding model, load it into Chroma, and then query it.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Chroma Multi-Modal Demo with LlamaIndex")

# %pip install llama-index-vector-stores-qdrant
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-vector-stores-chroma

# !pip install llama-index

"""
#### Creating a Chroma Index
"""
logger.info("#### Creating a Chroma Index")

# !pip install llama-index chromadb --quiet
# !pip install chromadb==0.4.17
# !pip install sentence-transformers
# !pip install pydantic==1.10.11
# !pip install open-clip-torch



# OPENAI_API_KEY = ""
# openai.api_key = OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""
## Download Images and Texts from Wikipedia
"""
logger.info("## Download Images and Texts from Wikipedia")



def get_wikipedia_images(title):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|dimensions|mime",
            "generator": "images",
            "gimlimit": "50",
        },
    ).json()
    image_urls = []
    for page in response["query"]["pages"].values():
        if page["imageinfo"][0]["url"].endswith(".jpg") or page["imageinfo"][
            0
        ]["url"].endswith(".png"):
            image_urls.append(page["imageinfo"][0]["url"])
    return image_urls


image_uuid = 0
MAX_IMAGES_PER_WIKI = 20

wiki_titles = {
    "Tesla Model X",
    "Pablo Picasso",
    "Rivian",
    "The Lord of the Rings",
    "The Matrix",
    "The Simpsons",
}

data_path = Path("mixed_wiki")
if not data_path.exists():
    Path.mkdir(data_path)

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

    images_per_wiki = 0
    try:
        list_img_urls = get_wikipedia_images(title)

        for url in list_img_urls:
            if url.endswith(".jpg") or url.endswith(".png"):
                image_uuid += 1

                urllib.request.urlretrieve(
                    url, data_path / f"{image_uuid}.jpg"
                )
                images_per_wiki += 1
                if images_per_wiki > MAX_IMAGES_PER_WIKI:
                    break
    except:
        logger.debug(str(Exception("No images found for Wikipedia page: ")) + title)
        continue

"""
## Set the embedding model
"""
logger.info("## Set the embedding model")


embedding_function = OpenCLIPEmbeddingFunction()

"""
## Build Chroma Multi-Modal Index with LlamaIndex
"""
logger.info("## Build Chroma Multi-Modal Index with LlamaIndex")


image_loader = ImageLoader()

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)


documents = SimpleDirectoryReader("./mixed_wiki/").load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

"""
## Retrieve results from Multi-Modal Index
"""
logger.info("## Retrieve results from Multi-Modal Index")

retriever = index.as_retriever(similarity_top_k=50)
retrieval_results = retriever.retrieve("Picasso famous paintings")



image_results = []
MAX_RES = 5
cnt = 0
for r in retrieval_results:
    if isinstance(r.node, ImageNode):
        image_results.append(r.node.metadata["file_path"])
    else:
        if cnt < MAX_RES:
            display_source_node(r)
        cnt += 1

display_image_uris(image_results, [3, 3], top_k=2)

logger.info("\n\n[DONE]", bright=True)