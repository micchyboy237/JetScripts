from IPython.display import HTML, display
from PIL import Image
from PIL import Image as _PILImage
from io import BytesIO
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_nomic import OllamaEmbeddings
from operator import itemgetter
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import base64
import chromadb
import io
import numpy as np
import os
import shutil
import uuid


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
## Nomic multi-modal RAG

Many documents contain a mixture of content types, including text and images. 

Yet, information captured in images is lost in most RAG applications.

With the emergence of multimodal LLMs, like [GPT-4V](https://ollama.com/research/gpt-4v-system-card), it is worth considering how to utilize images in RAG:

In this demo we

* Use multimodal embeddings from Nomic Embed [Vision](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) and [Text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) to embed images and text
* Retrieve both using similarity search
* Pass raw images and text chunks to a multimodal LLM for answer synthesis 

## Signup

Get your API token, then run:
```
! nomic login
```

Then run with your generated API token 
```
! nomic login < token > 
```

## Packages

For `unstructured`, you will also need `poppler` ([installation instructions](https://pdf2image.readthedocs.io/en/latest/installation.html)) and `tesseract` ([installation instructions](https://tesseract-ocr.github.io/tessdoc/Installation.html)) in your system.
"""
logger.info("## Nomic multi-modal RAG")

# ! nomic login token

# ! pip install -U langchain-nomic langchain-chroma langchain-community tiktoken langchain-ollama langchain # (newest versions required for multi-modal)

# ! pip install "unstructured[all-docs]==0.10.19" pillow pydantic lxml pillow matplotlib tiktoken

"""
## Data Loading

### Partition PDF text and images
  
Let's look at an example pdfs containing interesting images.

1/ Art from the J Paul Getty museum:

 * Here is a [zip file](https://drive.google.com/file/d/18kRKbq2dqAhhJ3DfZRnYcTBEUfYxe1YR/view?usp=sharing) with the PDF and the already extracted images. 
* https://www.getty.edu/publications/resources/virtuallibrary/0892360224.pdf

2/ Famous photographs from library of congress:

* https://www.loc.gov/lcm/pdf/LCM_2020_1112.pdf
* We'll use this as an example below

We can use `partition_pdf` below from [Unstructured](https://unstructured-io.github.io/unstructured/introduction.html#key-concepts) to extract text and images.

To supply this to extract the images:
```
extract_images_in_pdf=True
```



If using this zip file, then you can simply process the text only with:
```
extract_images_in_pdf=False
```
"""
logger.info("## Data Loading")


path = Path("../art")

path.resolve()


raw_pdf_elements = partition_pdf(
    filename=str(path.resolve()) + "/getty.pdf",
    extract_images_in_pdf=False,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))

"""
## Multi-modal embeddings with our document

We will use [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) embeddings. This model is aligned 
to [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) allowing for multimodal semantic search and Multimodal RAG!
"""
logger.info("## Multi-modal embeddings with our document")



text_vectorstore = Chroma(
    collection_name="mm_rag_clip_photos_text",
    embedding_function=OllamaEmbeddings(
        vision_model="nomic-embed-vision-v1.5", model="nomic-embed-text-v1.5"
    ),
)
image_vectorstore = Chroma(
    collection_name="mm_rag_clip_photos_image",
    embedding_function=OllamaEmbeddings(
        vision_model="nomic-embed-vision-v1.5", model="nomic-embed-text-v1.5"
    ),
)

image_uris = sorted(
    [
        os.path.join(path, image_name)
        for image_name in os.listdir(path)
        if image_name.endswith(".jpg")
    ]
)

image_vectorstore.add_images(uris=image_uris)

text_vectorstore.add_texts(texts=texts)

image_retriever = image_vectorstore.as_retriever()
text_retriever = text_vectorstore.as_retriever()

"""
## RAG

`vectorstore.add_images` will store / retrieve images as base64 encoded strings.

These can be passed to [GPT-4V](https://platform.ollama.com/docs/guides/vision).
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
    img = Image.open(io.BytesIO(img_data))

    resized_img = img.resize(size, Image.LANCZOS)

    buffered = io.BytesIO()
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
* Then we pass this into our prompt template, which is a custom function that formats the message for the gpt-4-vision-preview model. 
* And finally we parse the output as a string.
"""
logger.info("Currently, we format the inputs using a `RunnableLambda` while we add image support to `ChatPromptTemplates`.")


# os.environ["OPENAI_API_KEY"] = ""




def prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["text_context"]["texts"])
    messages = []

    if data_dict["image_context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['image_context']['images'][0]}"
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


model = ChatOllama(model="llama3.2")

chain = (
    {
        "text_context": text_retriever | RunnableLambda(split_image_text_types),
        "image_context": image_retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)

"""
## Test retrieval and run RAG
"""
logger.info("## Test retrieval and run RAG")



def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    display(HTML(image_html))


docs = text_retriever.invoke("Women with children", k=5)
for doc in docs:
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
    else:
        logger.debug(doc.page_content)

docs = image_retriever.invoke("Women with children", k=5)
for doc in docs:
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
    else:
        logger.debug(doc.page_content)

chain.invoke("Women with children")

"""
We can see the images retrieved in the LangSmith trace:

LangSmith [trace](https://smith.langchain.com/public/69c558a5-49dc-4c60-a49b-3adbb70f74c5/r/e872c2c8-528c-468f-aefd-8b5cd730a673).
"""
logger.info("We can see the images retrieved in the LangSmith trace:")

logger.info("\n\n[DONE]", bright=True)