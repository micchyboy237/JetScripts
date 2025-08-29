from PIL import Image
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pathlib import Path
import matplotlib.pyplot as plt
import os
import qdrant_client
import requests
import shutil
import time
import urllib.request
import wikipedia


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/multi_modal_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Modal RAG using Nomic Embed and Anthropic.

In this notebook, we show how to build a Multi-Modal RAG system using LlamaIndex, Nomic Embed, and Anthropic.

Wikipedia Text embedding index: [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

Wikipedia Images embedding index: [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)


Query encoder:
* Encoder query text for text index using Nomic Embed Text
* Encoder query text for image index using Nomic Embed Vision

Framework: [LlamaIndex](https://github.com/run-llama/llama_index)

Steps:
1. Download texts and images raw files for Wikipedia articles
2. Build text index for vector store using Nomic Embed Text embeddings
3. Build image index for vector store using Nomic Embed Vision embeddings
4. Retrieve relevant text and image simultaneously using different query encoding embeddings and vector stores
5. Pass retrieved texts and images to Claude 3
"""
logger.info("# Multi-Modal RAG using Nomic Embed and Anthropic.")

# %pip install llama-index-vector-stores-qdrant llama-index-multi-modal-llms-anthropic llama-index-embeddings-nomic

# %pip install llama_index ftfy regex tqdm
# %pip install matplotlib scikit-image
# %pip install -U qdrant_client
# %pip install wikipedia

"""
## Load and Download Multi-Modal datasets including texts and images from Wikipedia
Parse wikipedia articles and save into local folder
"""
logger.info(
    "## Load and Download Multi-Modal datasets including texts and images from Wikipedia")


wiki_titles = [
    "batman",
    "Vincent van Gogh",
    "San Francisco",
    "iPhone",
    "Tesla Model S",
    "BTS",
]


data_path = Path("data_wiki")

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

    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

"""
## Parse Wikipedia Images and texts. Load into local folder
"""
logger.info("## Parse Wikipedia Images and texts. Load into local folder")


image_path = Path("data_wiki")
image_uuid = 0
image_metadata_dict = {}
MAX_IMAGES_PER_WIKI = 30

wiki_titles = [
    "San Francisco",
    "Batman",
    "Vincent van Gogh",
    "iPhone",
    "Tesla Model S",
    "BTS band",
]

if not image_path.exists():
    Path.mkdir(image_path)


for title in wiki_titles:
    images_per_wiki = 0
    logger.debug(title)
    try:
        page_py = wikipedia.page(title)
        list_img_urls = page_py.images
        for url in list_img_urls:
            if url.endswith(".jpg") or url.endswith(".png"):
                image_uuid += 1
                image_file_name = title + "_" + url.split("/")[-1]

                image_metadata_dict[image_uuid] = {
                    "filename": image_file_name,
                    "img_path": "./" + str(image_path / f"{image_uuid}.jpg"),
                }

                req = urllib.request.Request(
                    url,
                    data=None,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36"
                    },
                )

                with urllib.request.urlopen(req) as response, open(
                    image_path / f"{image_uuid}.jpg", "wb"
                ) as out_file:
                    out_file.write(response.read())

                images_per_wiki += 1
                if images_per_wiki > MAX_IMAGES_PER_WIKI:
                    break

                time.sleep(1)  # Adjust the delay as needed

    except Exception as e:
        logger.debug(e)
        logger.debug(f"{images_per_wiki=}")
        continue


os.environ["NOMIC_API_KEY"] = ""
# os.environ["ANTHROPIC_API_KEY"] = ""

"""
## Build Multi Modal Vector Store using Text and Image embeddings under different collections
"""
logger.info(
    "## Build Multi Modal Vector Store using Text and Image embeddings under different collections")


client = qdrant_client.QdrantClient(path="qdrant_db")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)
embedding_model = NomicEmbedding(
    model_name="nomic-embed-text-v1.5",
    vision_model_name="nomic-embed-vision-v1.5",
)

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp_wiki/").load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embedding_model,
    image_embed_model=embedding_model,
)

"""
### Plot downloaded Images from Wikipedia
"""
logger.info("### Plot downloaded Images from Wikipedia")


def plot_images(image_metadata_dict):
    original_images_urls = []
    images_shown = 0
    for image_id in image_metadata_dict:
        img_path = image_metadata_dict[image_id]["img_path"]
        if os.path.isfile(img_path):
            filename = image_metadata_dict[image_id]["filename"]
            image = Image.open(img_path).convert("RGB")

            plt.subplot(9, 9, len(original_images_urls) + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            original_images_urls.append(filename)
            images_shown += 1
            if images_shown >= 81:
                break

    plt.tight_layout()


plot_images(image_metadata_dict)


def plot_images(image_paths):
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


"""
## Get Multi-Modal retrieval results for some example queries
"""
logger.info("## Get Multi-Modal retrieval results for some example queries")

test_query = "Who are the band members in BTS?"
retriever = index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)
retrieval_results = retriever.retrieve(test_query)


retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)

test_query = "What are Vincent van Gogh's famous paintings"
retriever = index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)
retrieval_results = retriever.retrieve(test_query)

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)

test_query = "What are the popular tourist attraction in San Francisco"
retriever = index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)
retrieval_results = retriever.retrieve(test_query)

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)

test_query = "Which company makes Tesla"
retriever = index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)
retrieval_results = retriever.retrieve(test_query)

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)

test_query = "what is the main character in Batman"
retriever = index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)
retrieval_results = retriever.retrieve(test_query)

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)

"""
## Multimodal RAG with Claude 3

Using Nomic Embed and Claude 3, we can now perform Multimodal RAG! The images and texts are passed to Claude 3 to reason over.
"""
logger.info("## Multimodal RAG with Claude 3")


query_engine = index.as_query_engine(
    llm=AnthropicMultiModal(), similarity_top_k=2, image_similarity_top_k=1
)

response = query_engine.query(
    "What are Vincent van Gogh's famous paintings and popular subjects?"
)

logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)
