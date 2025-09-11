from IPython.display import HTML, display
from PIL import Image
from base64 import b64decode
from io import BytesIO
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.smith import RunEvalConfig
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from operator import itemgetter
from unstructured.partition.pdf import partition_pdf
import base64
import io
import os
import pandas as pd
import re
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
# Advanced RAG Eval

The cookbook walks through the process of running eval(s) on advanced RAG. 

This can be very useful to determine the best RAG approach for your application.
"""
logger.info("# Advanced RAG Eval")

# ! pip install -U langchain ollama langchain_chroma langchain-experimental # (newest versions required for multi-modal)

# ! pip install "unstructured[all-docs]==0.10.19" pillow pydantic lxml matplotlib tiktoken open_clip_torch torch

"""
## Data Loading

Let's look at an [example whitepaper](https://sgp.fas.org/crs/misc/IF10244.pdf) that provides a mixture of tables, text, and images about Wildfires in the US.

### Option 1: Load text
"""
logger.info("## Data Loading")

path = "/Users/rlm/Desktop/cpi/"


loader = PyPDFLoader(path + "cpi.pdf")
pdf_pages = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits_pypdf = text_splitter.split_documents(pdf_pages)
all_splits_pypdf_texts = [d.page_content for d in all_splits_pypdf]

"""
### Option 2: Load text, tables, images
"""
logger.info("### Option 2: Load text, tables, images")


raw_pdf_elements = partition_pdf(
    filename=path + "cpi.pdf",
    extract_images_in_pdf=True,
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
## Store

### Option 1: Embed, store text chunks
"""
logger.info("## Store")


baseline = Chroma.from_texts(
    texts=all_splits_pypdf_texts,
    collection_name="baseline",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)
retriever_baseline = baseline.as_retriever()

"""
### Option 2: Multi-vector retriever

#### Text Summary
"""
logger.info("### Option 2: Multi-vector retriever")


prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
These summaries will be embedded and used to retrieve the raw text or table elements. \
Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

model = ChatOllama(model="llama3.2")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

"""
#### Image Summary
"""
logger.info("#### Image Summary")


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Image summary"""
    chat = ChatOllama(model="llama3.2")

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


img_base64_list = []

image_summaries = []

prompt = """You are an assistant tasked with summarizing images for retrieval. \
These summaries will be embedded and used to retrieve the raw image. \
Give a concise summary of the image that is well optimized for retrieval."""

for img_file in sorted(os.listdir(path)):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(path, img_file)
        base64_image = encode_image(img_path)
        img_base64_list.append(base64_image)
        image_summaries.append(image_summarize(base64_image, prompt))

"""
### Option 2a: Multi-vector retriever w/ raw images

* Return images to LLM for answer synthesis
"""
logger.info("### Option 2a: Multi-vector retriever w/ raw images")


def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


multi_vector_img = Chroma(
    collection_name="multi_vector_img", embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

retriever_multi_vector_img = create_multi_vector_retriever(
    multi_vector_img,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)

query = "What percentage of CPI is dedicated to Housing, and how does it compare to the combined percentage of Medical Care, Apparel, and Other Goods and Services?"
suffix_for_images = " Include any pie charts, graphs, or tables."
docs = retriever_multi_vector_img.invoke(query + suffix_for_images)


def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    display(HTML(image_html))


plt_img_base64(docs[1])

"""
### Option 2b: Multi-vector retriever w/ image summaries

* Return text summary of images to LLM for answer synthesis
"""
logger.info("### Option 2b: Multi-vector retriever w/ image summaries")

multi_vector_text = Chroma(
    collection_name="multi_vector_text", embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

retriever_multi_vector_img_summary = create_multi_vector_retriever(
    multi_vector_text,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)

"""
### Option 3: Multi-modal embeddings
"""
logger.info("### Option 3: Multi-modal embeddings")


multimodal_embd = Chroma(
    collection_name="multimodal_embd", embedding_function=OpenCLIPEmbeddings()
)

image_uris = sorted(
    [
        os.path.join(path, image_name)
        for image_name in os.listdir(path)
        if image_name.endswith(".jpg")
    ]
)

if image_uris:
    multimodal_embd.add_images(uris=image_uris)
if texts:
    multimodal_embd.add_texts(texts=texts)
if tables:
    multimodal_embd.add_texts(texts=tables)

retriever_multimodal_embd = multimodal_embd.as_retriever()

"""
## RAG

### Text Pipeline
"""
logger.info("## RAG")


template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
rag_prompt_text = ChatPromptTemplate.from_template(template)


def text_rag_chain(retriever):
    """RAG chain"""

    model = ChatOllama(model="llama3.2")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt_text
        | model
        | StrOutputParser()
    )

    return chain


"""
### Multi-modal Pipeline
"""
logger.info("### Multi-modal Pipeline")


def looks_like_base64(sb):
    """Check if the string looks like base64."""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """Check if the base64 data is an image by looking at the start of the data."""
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        # Decode and get the first 8 bytes
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def split_image_text_types(docs):
    """Split base64-encoded images and texts."""
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
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
            "Answer the question based only on the provided context, which can include text, tables, and image(s). "
            "If an image is provided, analyze it carefully to help answer the question.\n"
            f"User-provided question / keywords: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """Multi-modal RAG chain"""

    model = ChatOllama(model="llama3.2")

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain


"""
### Build RAG Pipelines
"""
logger.info("### Build RAG Pipelines")

chain_baseline = text_rag_chain(retriever_baseline)
chain_mv_text = text_rag_chain(retriever_multi_vector_img_summary)

chain_multimodal_mv_img = multi_modal_rag_chain(retriever_multi_vector_img)
chain_multimodal_embd = multi_modal_rag_chain(retriever_multimodal_embd)

"""
## Eval set
"""
logger.info("## Eval set")


eval_set = pd.read_csv(path + "cpi_eval.csv")
eval_set.head(3)


client = Client()
dataset_name = f"CPI Eval {str(uuid.uuid4())}"
dataset = client.create_dataset(dataset_name=dataset_name)

for _, row in eval_set.iterrows():
    q = row["Question"]
    a = row["Answer"]
    client.create_example(
        inputs={"question": q}, outputs={"answer": a}, dataset_id=dataset.id
    )


eval_config = RunEvalConfig(
    evaluators=["qa"],
)


def run_eval(chain, run_name, dataset_name):
    _ = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=lambda: (
            lambda x: x["question"] + suffix_for_images)
        | chain,
        evaluation=eval_config,
        project_name=run_name,
    )


for chain, run in zip(
    [chain_baseline, chain_mv_text, chain_multimodal_mv_img, chain_multimodal_embd],
    ["baseline", "mv_text", "mv_img", "mm_embd"],
):
    run_eval(chain, dataset_name + "-" + run, dataset_name)

logger.info("\n\n[DONE]", bright=True)
