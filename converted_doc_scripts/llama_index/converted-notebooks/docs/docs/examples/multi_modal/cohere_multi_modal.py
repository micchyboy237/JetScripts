from PIL import Image
from jet.logger import CustomLogger
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pathlib import Path
import matplotlib.pyplot as plt
import os
import qdrant_client
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/cohere_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Modal Retrieval using Cohere Multi-Modal Embeddings

[Cohere has released multi-modal embedding model](https://cohere.com/blog/multimodal-embed-3) and in this notebook, we will demonstrate `Multi-Modal Retrieval using Cohere MultiModal Embeddings`.

Why are MultiModal Embeddings important?

Multimodal embeddings are important because they allow AI systems to understand and search through both images and text in a unified way. Instead of having separate systems for text and image search, multimodal embeddings convert both types of content into the same embedding space, enabling users to find relevant information across different types of media for a given query.

For the demonstration, here are the steps:

1. Download text, images, and raw PDF files from related Wikipedia articles.
2. Build a Multi-Modal index for both texts and images using Cohere Multi-Modal Embeddings.
3. Retrieve relevant text and images simultaneously using a Multi-Modal Retriever for a query.
4. Generate responses using the Multi-Modal Query Engine for a query.

**NOTE:** We will use Anthropic's MultiModal LLM for response generation, as Cohere does not yet support MultiModal LLMs.

### Installation

We will use Cohere MultiModal embeddings for retrieval, Qdrant vector-store and Anthropic MultiModal LLM for response generation.
"""
logger.info("# Multi-Modal Retrieval using Cohere Multi-Modal Embeddings")

# %pip install llama-index-embeddings-cohere
# %pip install llama-index-vector-stores-qdrant
# %pip install llama-index-multi-modal-llms-anthropic

"""
### Setup API Keys

Cohere - MultiModal Retrieval

Anthropic - MultiModal LLM.
"""
logger.info("### Setup API Keys")


os.environ["COHERE_API_KEY"] = "<YOUR COHERE API KEY>"

# os.environ["ANTHROPIC_API_KEY"] = "<YOUR ANTHROPIC API KEY>"

"""
### Utils

1. `get_wikipedia_images`: Get the image URLs from the Wikipedia page with the specified title.
2. `plot_images`: Plot the images in the specified list of image paths.
3. `delete_large_images`: Delete images larger than 5 MB in the specified directory.

**NOTE**: Cohere API accepts images of size less than 5MB.
"""
logger.info("### Utils")



def get_wikipedia_images(title):
    """
    Get the image URLs from the Wikipedia page with the specified title.
    """
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


def plot_images(image_paths):
    """
    Plot the images in the specified list of image paths.
    """
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break


def delete_large_images(folder_path):
    """
    Delete images larger than 5 MB in the specified directory.
    """
    deleted_images = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp")
        ):
            file_path = os.path.join(folder_path, file_name)
            file_size = os.path.getsize(file_path)
            if file_size > 5242880:
                os.remove(file_path)
                deleted_images.append(file_name)
                logger.debug(
                    f"Image: {file_name} was larger than 5 MB and has been deleted."
                )

"""
### Donwload text and images from Wikipedia

We will download text and images associated from following wikipedia pages.

1. Audi e-tron
2. Ford Mustang
3. Porsche Taycan
"""
logger.info("### Donwload text and images from Wikipedia")

image_uuid = 0
image_metadata_dict = {}
MAX_IMAGES_PER_WIKI = 10

wiki_titles = {
    "Audi e-tron",
    "Ford Mustang",
    "Porsche Taycan",
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
            if (
                url.endswith(".jpg")
                or url.endswith(".png")
                or url.endswith(".svg")
            ):
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
### Delete larger image files

Cohere MultiModal Embedding model accepts less than 5MB file, so here we delete the larger image files.
"""
logger.info("### Delete larger image files")

delete_large_images(data_path)

"""
### Set Embedding Model and LLM.

Cohere MultiModal Embedding model for retrieval and Anthropic MultiModal LLM for response generation.
"""
logger.info("### Set Embedding Model and LLM.")


Settings.embed_model = CohereEmbedding(
    api_key=os.environ["COHERE_API_KEY"],
    model_name="embed-english-v3.0",  # current v3 models support multimodal embeddings
)

anthropic_multimodal_llm = AnthropicMultiModal(max_tokens=300)

"""
### Load the data

We will load the downloaded text and image data.
"""
logger.info("### Load the data")


documents = SimpleDirectoryReader("./mixed_wiki/").load_data()

"""
### Setup Qdrant VectorStore

We will use Qdrant vector-store for storing image and text embeddings and associated metadata.
"""
logger.info("### Setup Qdrant VectorStore")



client = qdrant_client.QdrantClient(path="qdrant_mm_db")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

"""
### Create MultiModalVectorStoreIndex.
"""
logger.info("### Create MultiModalVectorStoreIndex.")

index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    image_embed_model=Settings.embed_model,
)

"""
### Test the Retrieval

Here we create a retriever and test it out.
"""
logger.info("### Test the Retrieval")

retriever_engine = index.as_retriever(
    similarity_top_k=4, image_similarity_top_k=4
)

query = "Which models of Porsche are discussed here?"
retrieval_results = retriever_engine.retrieve(query)

"""
#### Inspect the retrieval results
"""
logger.info("#### Inspect the retrieval results")


retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)

"""
### Test the MultiModal QueryEngine

We will create a `QueryEngine` by using the above `MultiModalVectorStoreIndex`.
"""
logger.info("### Test the MultiModal QueryEngine")


qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(
    llm=anthropic_multimodal_llm, text_qa_template=qa_tmpl
)

query = "Which models of Porsche are discussed here?"
response = query_engine.query(query)

logger.debug(str(response))

"""
#### Inspect the sources
"""
logger.info("#### Inspect the sources")


for text_node in response.metadata["text_nodes"]:
    display_source_node(text_node, source_length=200)
plot_images(
    [n.metadata["file_path"] for n in response.metadata["image_nodes"]]
)

logger.info("\n\n[DONE]", bright=True)