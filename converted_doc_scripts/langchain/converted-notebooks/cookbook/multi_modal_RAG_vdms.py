from IPython.display import HTML, display
from PIL import Image
from io import BytesIO
from jet.adapters.langchain.chat_ollama.llms import OllamaLLM
from jet.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_vdms import VDMS
from langchain_vdms.vectorstores import VDMS_Client
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import base64
import os
import requests
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
## VDMS multi-modal RAG

Many documents contain a mixture of content types, including text and images. 

Yet, information captured in images is lost in most RAG applications.

With the emergence of multimodal LLMs, like [GPT-4V](https://ollama.com/research/gpt-4v-system-card), it is worth considering how to utilize images in RAG. 

This cookbook highlights: 
* Use of [Unstructured](https://unstructured.io/) to parse images, text, and tables from documents (PDFs).
* Use of multimodal embeddings (such as [CLIP](https://ollama.com/research/clip)) to embed images and text
* Use of [VDMS](https://github.com/IntelLabs/vdms/blob/master/README.md) as a vector store with support for multi-modal
* Retrieval of both images and text using similarity search
* Passing raw images and text chunks to a multimodal LLM for answer synthesis

## Packages

For `unstructured`, you will also need `poppler` ([installation instructions](https://pdf2image.readthedocs.io/en/latest/installation.html)) and `tesseract` ([installation instructions](https://tesseract-ocr.github.io/tessdoc/Installation.html)) in your system.
"""
logger.info("## VDMS multi-modal RAG")

# ! pip install --quiet -U langchain-vdms langchain-experimental langchain-ollama

# ! pip install --quiet pdf2image "unstructured[all-docs]==0.10.19" "onnxruntime==1.17.0" pillow pydantic lxml open_clip_torch


"""
## Start VDMS Server

Let's start a VDMS docker using port 55559 instead of default 55555. 
Keep note of the port and hostname as this is needed for the vector store as it uses the VDMS Python client to connect to the server.
"""
logger.info("## Start VDMS Server")

# ! docker run --rm -d -p 55559:55555 --name vdms_rag_nb intellabs/vdms:latest


vdms_client = VDMS_Client(port=55559)

"""
## Data Loading

### Partition PDF text and images
  
Let's use famous photographs from the PDF version of Library of Congress Magazine in this example.

We can use `partition_pdf` from [Unstructured](https://unstructured-io.github.io/unstructured/introduction.html#key-concepts) to extract text and images.
"""
logger.info("## Data Loading")



base_datapath = Path("./data/multimodal_files").resolve()
datapath = base_datapath / "images"
datapath.mkdir(parents=True, exist_ok=True)

pdf_url = "https://www.loc.gov/lcm/pdf/LCM_2020_1112.pdf"
pdf_path = str(base_datapath / pdf_url.split("/")[-1])
with open(pdf_path, "wb") as f:
    f.write(requests.get(pdf_url).content)


raw_pdf_elements = partition_pdf(
    filename=pdf_path,
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=datapath,
)

datapath = str(datapath)

tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))

"""
## Multi-modal embeddings with our document

In this section, we initialize the VDMS vector store for both text and images. For better performance, we use model `ViT-g-14` from [OpenClip multimodal embeddings](https://python.langchain.com/docs/integrations/text_embedding/open_clip).
The images are stored as base64 encoded strings with `vectorstore.add_images`.
"""
logger.info("## Multi-modal embeddings with our document")



vectorstore = VDMS(
    client=vdms_client,
    collection_name="mm_rag_clip_photos",
    embedding=OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k"),
)

image_uris = sorted(
    [
        os.path.join(datapath, image_name)
        for image_name in os.listdir(datapath)
        if image_name.endswith(".jpg")
    ]
)

if image_uris:
    vectorstore.add_images(uris=image_uris)

if texts:
    vectorstore.add_texts(texts=texts)

retriever = vectorstore.as_retriever()

"""
## RAG

Here we define helper functions for image results.
"""
logger.info("## RAG")




def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))

    resized_img = img.resize(size, Image.LANCZOS)

    buffered = BytesIO()
    resized_img.save(buffered, format=img.format)

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}

"""
Currently, we format the inputs using a `RunnableLambda` while we add image support to `ChatPromptTemplates`.

Our runnable follows the classic RAG flow - 

* We first compute the context (both "texts" and "images" in this case) and the question (just a RunnablePassthrough here) 
* Then we pass this into our prompt template, which is a custom function that formats the message for the llava model. 
* And finally we parse the output as a string.

Here we are using Ollama to serve the Llava model. Please see [Ollama](https://python.langchain.com/docs/integrations/llms/ollama) for setup instructions.
"""
logger.info("Currently, we format the inputs using a `RunnableLambda` while we add image support to `ChatPromptTemplates`.")



def prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

    text_message = {
        "type": "text",
        "text": (
            "As an expert art critic and historian, your task is to analyze and interpret images, "
            "considering their historical and cultural significance. Alongside the images, you will be "
            "provided with related text to offer context. Both will be retrieved from a vectorstore based "
            "on user-input keywords. Please use your extensive knowledge and analytical skills to provide a "
            "comprehensive summary that includes:\n"
            "- A detailed description of the visual elements in the image.\n"
            "- The historical and cultural context of the image.\n"
            "- An interpretation of the image's symbolism and meaning.\n"
            "- Connections between the image and the related text.\n\n"
            f"User-provided keywords: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """Multi-modal RAG chain"""

    llm_model = OllamaLLM(
        verbose=True, temperature=0.5, model="llava", base_url="http://localhost:11434"
    )

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(prompt_func)
        | llm_model
        | StrOutputParser()
    )

    return chain

"""
## Test retrieval and run RAG
Now let's query for a `woman with children` and retrieve the top results.
"""
logger.info("## Test retrieval and run RAG")



def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    display(HTML(image_html))


query = "Woman with children"
docs = retriever.invoke(query, k=10)

for doc in docs:
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
    else:
        logger.debug(doc.page_content)

"""
Now let's use the `multi_modal_rag_chain` to process the same query and display the response.
"""
logger.info("Now let's use the `multi_modal_rag_chain` to process the same query and display the response.")

chain = multi_modal_rag_chain(retriever)
response = chain.invoke(query)
logger.debug(response)

# ! docker kill vdms_rag_nb

"""

"""

logger.info("\n\n[DONE]", bright=True)