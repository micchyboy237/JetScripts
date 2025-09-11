from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
from typing import Any
from unstructured.partition.pdf import partition_pdf
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
## Semi-structured RAG

Many documents contain a mixture of content types, including text and tables. 

Semi-structured data can be challenging for conventional RAG for at least two reasons: 

* Text splitting may break up tables, corrupting the data in retrieval
* Embedding tables may pose challenges for semantic similarity search 

This cookbook shows how to perform RAG on documents with semi-structured data: 

* We will use [Unstructured](https://unstructured.io/) to parse both text and tables from documents (PDFs).
* We will use the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector) to store raw tables, text along with table summaries better suited for retrieval.
* We will use [LCEL](https://python.langchain.com/docs/expression_language/) to implement the chains used.

The overall flow is here:

![MVR.png](attachment:7b5c5a30-393c-4b27-8fa1-688306ef2aef.png)

## Packages
"""
logger.info("## Semi-structured RAG")

# ! pip install langchain langchain-chroma "unstructured[all-docs]" pydantic lxml langchainhub

"""
The PDF partitioning used by Unstructured will use: 

* `tesseract` for Optical Character Recognition (OCR)
*  `poppler` for PDF rendering and processing
"""
logger.info("The PDF partitioning used by Unstructured will use:")

# ! brew install tesseract
# ! brew install poppler

"""
## Data Loading

### Partition PDF tables and text

Apply to the [`LLaMA2`](https://arxiv.org/pdf/2307.09288.pdf) paper. 

We use the Unstructured [`partition_pdf`](https://unstructured-io.github.io/unstructured/core/partition.html#partition-pdf), which segments a PDF document by using a layout model. 

This layout model makes it possible to extract elements, such as tables, from pdfs. 

We also can use `Unstructured` chunking, which:

* Tries to identify document sections (e.g., Introduction, etc)
* Then, builds text blocks that maintain sections while also honoring user-defined chunk sizes
"""
logger.info("## Data Loading")

path = "/Users/rlm/Desktop/Papers/LLaMA2/"


raw_pdf_elements = partition_pdf(
    filename=path + "LLaMA2.pdf",
    extract_images_in_pdf=False,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

"""
We can examine the elements extracted by `partition_pdf`.

`CompositeElement` are aggregated chunks.
"""
logger.info("We can examine the elements extracted by `partition_pdf`.")

category_counts = {}

for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

unique_categories = set(category_counts.keys())
category_counts


class Element(BaseModel):
    type: str
    text: Any


categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

table_elements = [e for e in categorized_elements if e.type == "table"]
logger.debug(len(table_elements))

text_elements = [e for e in categorized_elements if e.type == "text"]
logger.debug(len(text_elements))

"""
## Multi-vector retriever

Use [multi-vector-retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector#summary) to produce summaries of tables and, optionally, text. 

With the summary, we will also store the raw table elements.

The summaries are used to improve the quality of retrieval, [as explained in the multi vector retriever docs](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector).

The raw tables are passed to the LLM, providing the full table context for the LLM to generate the answer.  

### Summaries
"""
logger.info("## Multi-vector retriever")


"""
We create a simple summarize chain for each element.

You can also see, re-use, or modify the prompt in the Hub [here](https://smith.langchain.com/hub/rlm/multi-vector-retriever-summarization).

```
obj = hub.pull("rlm/multi-vector-retriever-summarization")
```
"""
logger.info("We create a simple summarize chain for each element.")

prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

model = ChatOllama(model="llama3.2")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

"""
### Add to vectorstore

Use [Multi Vector Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector#summary) with summaries: 

* `InMemoryStore` stores the raw text, tables
* `vectorstore` stores the embedded summaries
"""
logger.info("### Add to vectorstore")


vectorstore = Chroma(collection_name="summaries",
                     embedding_function=OllamaEmbeddings(model="mxbai-embed-large"))

store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

"""
## RAG

Run [RAG pipeline](https://python.langchain.com/docs/expression_language/cookbook/retrieval).
"""
logger.info("## RAG")


template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3.2")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("What is the number of training tokens for LLaMA2?")

"""
We can check the [trace](https://smith.langchain.com/public/4739ae7c-1a13-406d-bc4e-3462670ebc01/r) to see what chunks were retrieved:

This includes Table 1 of the paper, showing the Tokens used for training.

```
Training Data Params Context GQA Tokens LR Length 7B 2k 1.0T 3.0x 10-4 See Touvron et al. 13B 2k 1.0T 3.0 x 10-4 LiaMa 1 (2023) 33B 2k 14T 1.5 x 10-4 65B 2k 1.4T 1.5 x 10-4 7B 4k 2.0T 3.0x 10-4 Liama 2 A new mix of publicly 13B 4k 2.0T 3.0 x 10-4 available online data 34B 4k v 2.0T 1.5 x 10-4 70B 4k v 2.0T 1.5 x 10-4
```
"""
logger.info(
    "We can check the [trace](https://smith.langchain.com/public/4739ae7c-1a13-406d-bc4e-3462670ebc01/r) to see what chunks were retrieved:")

logger.info("\n\n[DONE]", bright=True)
