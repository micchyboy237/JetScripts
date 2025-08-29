from PIL import Image
from jet.logger import CustomLogger
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OllamaFunctionCallingAdapterMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pathlib import Path
import matplotlib.pyplot as plt
import os
import qdrant_client
import shutil
import urllib.request
import wikipedia


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/image_to_image_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Image to Image Retrieval using CLIP embedding and image correlation reasoning using GPT4V

In this notebook, we show how to build a Image to Image retrieval using LlamaIndex with GPT4-V and CLIP.

LlamaIndex Image to Image Retrieval 

- Images embedding index: [CLIP](https://github.com/openai/CLIP) embeddings from OllamaFunctionCallingAdapter for images


Framework: [LlamaIndex](https://github.com/run-llama/llama_index)

Steps:
1. Download texts, images, pdf raw files from Wikipedia pages

2. Build Multi-Modal index and vetor store for both texts and images

3. Retrieve relevant images given a image query using Multi-Modal Retriever

4. Using GPT4V for reasoning the correlations between the input image and retrieved images
"""
logger.info("# Image to Image Retrieval using CLIP embedding and image correlation reasoning using GPT4V")

# %pip install llama-index-multi-modal-llms-openai
# %pip install llama-index-vector-stores-qdrant

# %pip install llama_index ftfy regex tqdm
# %pip install git+https://github.com/openai/CLIP.git
# %pip install torch torchvision
# %pip install matplotlib scikit-image
# %pip install -U qdrant_client


# OPENAI_API_KEY = "sk-"
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""
## Download images and texts from Wikipedia
"""
logger.info("## Download images and texts from Wikipedia")



image_path = Path("mixed_wiki")
image_uuid = 0
image_metadata_dict = {}
MAX_IMAGES_PER_WIKI = 30

wiki_titles = [
    "Vincent van Gogh",
    "San Francisco",
    "Batman",
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
                urllib.request.urlretrieve(
                    url, image_path / f"{image_uuid}.jpg"
                )
                images_per_wiki += 1
                if images_per_wiki > MAX_IMAGES_PER_WIKI:
                    break
    except:
        logger.debug(str(Exception("No images found for Wikipedia page: ")) + title)
        continue

"""
### Plot images from Wikipedia
"""
logger.info("### Plot images from Wikipedia")


image_paths = []
for img_path in os.listdir("./mixed_wiki"):
    image_paths.append(str(os.path.join("./mixed_wiki", img_path)))


def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break


plot_images(image_paths)

"""
## Build Multi-Modal index and Vector Store to index both text and images from Wikipedia
"""
logger.info("## Build Multi-Modal index and Vector Store to index both text and images from Wikipedia")




client = qdrant_client.QdrantClient(path="qdrant_img_db")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

documents = SimpleDirectoryReader("./mixed_wiki/").load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

"""
## Plot input query image
"""
logger.info("## Plot input query image")

input_image = "./mixed_wiki/2.jpg"
plot_images([input_image])

"""
## Retrieve images from Multi-Modal Index given the image query

### 1. Image to Image Retrieval Results
"""
logger.info("## Retrieve images from Multi-Modal Index given the image query")

retriever_engine = index.as_retriever(image_similarity_top_k=4)
retrieval_results = retriever_engine.image_to_image_retrieve(
    "./mixed_wiki/2.jpg"
)
retrieved_images = []
for res in retrieval_results:
    retrieved_images.append(res.node.metadata["file_path"])

plot_images(retrieved_images[1:])

"""
### 2. GPT4V Reasoning Retrieved Images based on Input Image
"""
logger.info("### 2. GPT4V Reasoning Retrieved Images based on Input Image")


image_documents = [ImageDocument(image_path=input_image)]

for res_img in retrieved_images[1:]:
    image_documents.append(ImageDocument(image_path=res_img))


openai_mm_llm = OllamaFunctionCallingAdapterMultiModal(
#     model="llama3.2", request_timeout=300.0, context_window=4096, api_key=OPENAI_API_KEY, max_new_tokens=1500
)
response = openai_mm_llm.complete(
    prompt="Given the first image as the base image, what the other images correspond to?",
    image_documents=image_documents,
)

logger.debug(response)

"""
## Using Image Query Engine 

Inside Query Engine, there are few steps:

1. Retrieve relevant images based on input image

2. Compose the `image_qa_template`` by using the promt text

3. Sending top k retrieved images and image_qa_template for GPT4V to answer/synthesis
"""
logger.info("## Using Image Query Engine")



qa_tmpl_str = (
    "Given the images provided, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_tmpl = PromptTemplate(qa_tmpl_str)


openai_mm_llm = OllamaFunctionCallingAdapterMultiModal(
#     model="llama3.2", request_timeout=300.0, context_window=4096, api_key=OPENAI_API_KEY, max_new_tokens=1500
)

query_engine = index.as_query_engine(
    llm=openai_mm_llm, image_qa_template=qa_tmpl
)

query_str = "Tell me more about the relationship between those paintings. "
response = query_engine.image_query("./mixed_wiki/2.jpg", query_str)

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)