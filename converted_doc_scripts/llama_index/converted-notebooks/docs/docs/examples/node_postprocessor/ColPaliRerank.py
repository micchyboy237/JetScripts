from PIL import Image
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core.base.response.schema import Response
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import (
CustomQueryEngine,
SimpleMultiModalQueryEngine,
)
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
ImageNode,
NodeWithScore,
MetadataMode,
TextNode,
)
from llama_index.core.schema import ImageNode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.multi_modal_llms.openai import MLXMultiModal
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.colpali_rerank import ColPaliRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pathlib import Path
from typing import Optional
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/colpalirerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Reranking using ColPali, Cohere Reranker and Multi-Modal Embeddings

[ColPali](https://huggingface.co/vidore/colpali-v1.2): ColPali it is a model based on a novel model architecture and training strategy based on Vision Language Models (VLMs), to efficiently index documents from their visual features.

In this notebook, we will demonstrate the usage of ColPali as a reranker on top of retrieved images using Cohere Multi-Modal embeddings.

For the demonstration, here are the steps:

1. Download text, images, and raw PDF files from related Wikipedia articles.
2. Build a Multi-Modal index for both texts and images using Cohere Multi-Modal Embeddings.
3. Retrieve relevant text and images simultaneously using a Multi-Modal Retriever for a query.
4. Rerank text nodes using Cohere Reranker and image nodes using ColPali.
5. Generate responses using the Multi-Modal Query Engine for a query using qwen3-1.7b-4bit Multi-Modal LLM.

### Installation

We will use Cohere MultiModal embeddings for retrieval, ColPali as reranker for image nodes, Cohere reranker for text nodes, Qdrant vector-store and MLX MultiModal LLM for response generation.
"""
logger.info("# Reranking using ColPali, Cohere Reranker and Multi-Modal Embeddings")

# %pip install llama-index-postprocessor-colpali-rerank
# %pip install llama-index-postprocessor-cohere-rerank
# %pip install llama-index-embeddings-cohere
# %pip install llama-index-vector-stores-qdrant
# %pip install llama-index-multi-modal-llms-openai

"""
### Setup API Keys

Cohere - MultiModal Retrieval

MLX - MultiModal LLM.
"""
logger.info("### Setup API Keys")


os.environ["COHERE_API_KEY"] = "<YOUR COHERE API KEY>"

# os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"

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

Cohere MultiModal Embedding model for retrieval and MLX MultiModal LLM for response generation.
"""
logger.info("### Set Embedding Model and LLM.")


Settings.embed_model = CohereEmbedding(
    api_key=os.environ["COHERE_API_KEY"],
    model_name="embed-english-v3.0",  # current v3 models support multimodal embeddings
)

gpt_4o = MLXMultiModal(model="qwen3-1.7b-4bit", max_new_tokens=4096)

"""
### Setup Cohere Reranker
"""
logger.info("### Setup Cohere Reranker")


cohere_rerank = CohereRerank(api_key=os.environ["COHERE_API_KEY"], top_n=3)

"""
### Setup ColPali Reranker
"""
logger.info("### Setup ColPali Reranker")


colpali_reranker = ColPaliRerank(
    top_n=3,
    model="vidore/colpali-v1.2",
    keep_retrieval_score=True,
    device="cuda",  # or "cpu" or "cuda:0" or "mps" for Apple
)

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
    similarity_top_k=6, image_similarity_top_k=6
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


QA_PROMPT_TMPL = """\
Below we give parsed text and images as context.

Use both the parsed text and images to answer the question.

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
from the text or image, and if there's discrepancies, and your reasoning for the final answer.

Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)


class MultimodalQueryEngine(CustomQueryEngine):
    """Custom multimodal Query Engine.

    Takes in a retriever to retrieve a set of document nodes.
    Also takes in a prompt template and multimodal model.

    """

    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: MLXMultiModal

    def __init__(
        self, qa_prompt: Optional[PromptTemplate] = None, **kwargs
    ) -> None:
        """Initialize."""
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        image_nodes = [n for n in nodes if isinstance(n.node, ImageNode)]
        text_nodes = [n for n in nodes if isinstance(n.node, TextNode)]

        query_bundle = QueryBundle(query_str)

        reranked_text_nodes = cohere_rerank.postprocess_nodes(
            text_nodes, query_bundle
        )

        reranked_image_nodes = colpali_reranker.postprocess_nodes(
            image_nodes, query_bundle
        )

        context_str = "\n\n".join(
            [
                r.get_content(metadata_mode=MetadataMode.LLM)
                for r in reranked_text_nodes
            ]
        )
        fmt_prompt = self.qa_prompt.format(
            context_str=context_str, query_str=query_str
        )

        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[n.node for n in reranked_image_nodes],
        )
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={
                "text_nodes": reranked_text_nodes,
                "image_nodes": reranked_image_nodes,
            },
        )

        return response

query_engine = MultimodalQueryEngine(
    retriever=retriever_engine, multi_modal_llm=gpt_4o
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