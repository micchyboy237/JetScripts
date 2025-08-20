from PIL import Image
from jet.logger import CustomLogger
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.core.response.notebook_utils import (
display_source_node,
display_image_uris,
)
from llama_index.core.schema import ImageDocument
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.multi_modal_llms.replicate.base import (
REPLICATE_MULTI_MODAL_LLM_MODELS,
)
from llama_index.readers.file import FlatReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pathlib import Path
from pydantic import BaseModel
import matplotlib.pyplot as plt
import openai
import os
import pickle
import qdrant_client
import requests
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/llava_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# LlaVa Demo with LlamaIndex

In this example, we illustrate how we use LlaVa for belowing tasks:
* Retrieval Augmented Image Captioning
* Pydantic Structured Output
* Multi-Modal Retrieval-Augmented Generation (RAG) using Llava-13b

Context for LLaVA: Large Language and Vision Assistant
* [Website](https://llava-vl.github.io/)
* [Paper](https://arxiv.org/abs/2304.08485)
* [Github](https://github.com/haotian-liu/LLaVA)
* LLaVA 13b is now supported in Replicate: [See here.](https://replicate.com/yorickvp/llava-13b)

For LlamaIndex:
LlaVa+Replicate enables us to run image understanding locally and combine the multi-modal knowledge with our RAG knowledge based system.

## Retrieval Augmented Image Captioning using Llava-13b
### Using Replicate serving LLaVa model through LlamaIndex
"""
logger.info("# LlaVa Demo with LlamaIndex")

# %pip install llama-index-vector-stores-qdrant
# %pip install llama-index-readers-file
# %pip install llama-index-multi-modal-llms-replicate

# %pip install unstructured replicate
# %pip install llama_index ftfy regex tqdm
# %pip install git+https://github.com/openai/CLIP.git
# %pip install torch torchvision
# %pip install matplotlib scikit-image
# %pip install -U qdrant_client


REPLICATE_API_TOKEN = "..."  # Your Relicate API token here
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

"""
## Perform Data Extraction from Tesla 10K file


In these sections we use Unstructured to parse out the table and non-table elements.

### Extract Elements

We use Unstructured to extract table and non-table elements from the 10-K filing.
"""
logger.info("## Perform Data Extraction from Tesla 10K file")

# !wget "https://www.dropbox.com/scl/fi/mlaymdy1ni1ovyeykhhuk/tesla_2021_10k.htm?rlkey=qf9k4zn0ejrbm716j0gg7r802&dl=1" -O tesla_2021_10k.htm
# !wget "https://docs.google.com/uc?export=download&id=1UU0xc3uLXs-WG0aDQSXjGacUkp142rLS" -O texas.jpg


reader = FlatReader()
docs_2021 = reader.load_data(Path("tesla_2021_10k.htm"))
node_parser = UnstructuredElementNodeParser()


# OPENAI_API_KEY = "..."
# openai.api_key = OPENAI_API_KEY  # add your openai api key here
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


if not os.path.exists("2021_nodes.pkl"):
    raw_nodes_2021 = node_parser.get_nodes_from_documents(docs_2021)
    pickle.dump(raw_nodes_2021, open("2021_nodes.pkl", "wb"))
else:
    raw_nodes_2021 = pickle.load(open("2021_nodes.pkl", "rb"))

nodes_2021, objects_2021 = node_parser.get_nodes_and_objects(raw_nodes_2021)

"""
## Setup Composable Retriever

Now that we've extracted tables and their summaries, we can setup a composable retriever in LlamaIndex to query these tables.

### Construct Retrievers
"""
logger.info("## Setup Composable Retriever")


vector_index = VectorStoreIndex(nodes=nodes_2021, objects=objects_2021)
query_engine = vector_index.as_query_engine(similarity_top_k=5, verbose=True)


imageUrl = "./texas.jpg"
image = Image.open(imageUrl).convert("RGB")

plt.figure(figsize=(16, 5))
plt.imshow(image)

"""
### Running LLaVa model using Replicate through LlamaIndex for image understanding
"""
logger.info("### Running LLaVa model using Replicate through LlamaIndex for image understanding")


logger.debug(imageUrl)

llava_multi_modal_llm = ReplicateMultiModal(
    model=REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"],
    max_new_tokens=200,
    temperature=0.1,
)

prompt = "which Tesla factory is shown in the image? Please answer just the name of the factory."

llava_response = llava_multi_modal_llm.complete(
    prompt=prompt,
    image_documents=[ImageDocument(image_path=imageUrl)],
)

logger.debug(llava_response.text)

"""
### Retrieve relevant information from LlamaIndex knowledge base based on LLaVa image understanding to augment `Image Captioning`
"""
logger.info("### Retrieve relevant information from LlamaIndex knowledge base based on LLaVa image understanding to augment `Image Captioning`")

rag_response = query_engine.query(llava_response.text)

logger.debug(rag_response)

"""
## Multi-Modal Pydantic Program with LLaVa

### Initialize the Instagram Ads Pydantic Class
"""
logger.info("## Multi-Modal Pydantic Program with LLaVa")

input_image_path = Path("instagram_images")
if not input_image_path.exists():
    Path.mkdir(input_image_path)

# !wget "https://docs.google.com/uc?export=download&id=12ZpBBFkYu-jzz1iz356U5kMikn4uN9ww" -O ./instagram_images/jordan.png



class InsAds(BaseModel):
    """Data model for a Ins Ads."""

    account: str
    brand: str
    product: str
    category: str
    discount: str
    price: str
    comments: str
    review: str
    description: str


ins_imageUrl = "./instagram_images/jordan.png"
image = Image.open(ins_imageUrl).convert("RGB")

plt.figure(figsize=(16, 5))
plt.imshow(image)

"""
### Using Multi-Modal Pydantic Program to generate structured output using Llava-13b
"""
logger.info("### Using Multi-Modal Pydantic Program to generate structured output using Llava-13b")


prompt_template_str = """\
    can you summarize what is in the image\
    and return the answer with json format \
"""


def pydantic_llava(
    model_name, output_class, image_documents, prompt_template_str
):
    mm_llm = ReplicateMultiModal(
        model=REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"],
        max_new_tokens=1000,
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=mm_llm,
        verbose=True,
    )

    response = llm_program()
    logger.debug(f"Model: {model_name}")
    for res in response:
        logger.debug(res)
    return response

"""
### Output Structured Pydantic Output
"""
logger.info("### Output Structured Pydantic Output")


ins_image_documents = SimpleDirectoryReader("./instagram_images").load_data()

pydantic_response = pydantic_llava(
    "llava-13b", InsAds, ins_image_documents, prompt_template_str
)

logger.debug(pydantic_response.brand)

"""
## Advanced Multi-Modal Retrieval using GPT4V and Multi-Modal Index/Retriever/Query Engine

### Downloading text, images data from raw files [Wikipedia] for Multi Modal Index/Retrieval
"""
logger.info("## Advanced Multi-Modal Retrieval using GPT4V and Multi-Modal Index/Retriever/Query Engine")


wiki_titles = [
    "batman",
    "Vincent van Gogh",
    "San Francisco",
    "iPhone",
    "Tesla Model S",
    "BTS",
    "Air Jordan",
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


image_path = Path("data_wiki")
image_uuid = 0
image_metadata_dict = {}
MAX_IMAGES_PER_WIKI = 30

wiki_titles = [
    "Air Jordan",
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
### Build Multi-modal index and Vector Store to index both text and images
"""
logger.info("### Build Multi-modal index and Vector Store to index both text and images")


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

documents = SimpleDirectoryReader("./data_wiki/").load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)



def plot_images(image_metadata_dict):
    original_images_urls = []
    images_shown = 0
    for image_id in image_metadata_dict:
        img_path = image_metadata_dict[image_id]["img_path"]
        if os.path.isfile(img_path):
            filename = image_metadata_dict[image_id]["filename"]
            image = Image.open(img_path).convert("RGB")

            plt.subplot(8, 8, len(original_images_urls) + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            original_images_urls.append(filename)
            images_shown += 1
            if images_shown >= 64:
                break

    plt.tight_layout()


plot_images(image_metadata_dict)

"""
### Multi-Modal RAG Retrieval and Querying using LlaVa pydantic structured output
"""
logger.info("### Multi-Modal RAG Retrieval and Querying using LlaVa pydantic structured output")

retriever = index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)
retrieval_results = retriever.retrieve(pydantic_response.brand)


retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)


display_image_uris(retrieved_image)

"""
### Synthesis the RAG results using retrieved texts and images
"""
logger.info("### Synthesis the RAG results using retrieved texts and images")


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
    llm=llava_multi_modal_llm,
    text_qa_template=qa_tmpl,
    similarity_top_k=2,
    image_similarity_top_k=1,
)

query_str = "Tell me more about the " + pydantic_response.brand + " brand."
response = query_engine.query(query_str)

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)