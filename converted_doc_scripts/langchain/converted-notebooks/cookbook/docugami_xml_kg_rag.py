from dgml_utils.segmentation import get_chunks_str
from docugami import Docugami
from docugami.lib.upload import upload_to_named_docset, wait_for_dgml
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
from pprint import pprint
import os
import requests
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
## Docugami RAG over XML Knowledge Graphs (KG-RAG)

Many documents contain a mixture of content types, including text and tables. 

Semi-structured data can be challenging for conventional RAG for a few reasons since semantics may be lost by text-only chunking techniques, e.g.: 

* Text splitting may break up tables, corrupting the data in retrieval
* Embedding tables may pose challenges for semantic similarity search 

Docugami deconstructs documents into XML Knowledge Graphs consisting of hierarchical semantic chunks using the XML data model. This cookbook shows how to perform RAG using XML Knowledge Graphs as input (**KG-RAG**):

* We will use [Docugami](http://docugami.com/) to segment out text and table chunks from documents (PDF \[scanned or digital\], DOC or DOCX) including semantic XML markup in the chunks.
* We will use the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector) to store raw tables and text (including semantic XML markup) along with table summaries better suited for retrieval.
* We will use [LCEL](https://python.langchain.com/docs/expression_language/) to implement the chains used.

The overall flow is here:

![image.png](attachment:image.png)

## Packages
"""
logger.info("## Docugami RAG over XML Knowledge Graphs (KG-RAG)")

# ! pip install langchain docugami==0.0.8 dgml-utils==0.3.0 pydantic langchainhub langchain-chroma hnswlib --upgrade --quiet

"""
Docugami processes documents in the cloud, so you don't need to install any additional local dependencies.

## Data Loading

Let's use Docugami to process some documents. Here's what you need to get started:

1. Create a [Docugami workspace](http://www.docugami.com) (free trials available)
1. Create an access token via the Developer Playground for your workspace. [Detailed instructions](https://help.docugami.com/home/docugami-api).
1. Add your documents (PDF \[scanned or digital\], DOC or DOCX) to Docugami for processing. There are two ways to do this:
    1. Use the simple Docugami web experience. [Detailed instructions](https://help.docugami.com/home/adding-documents).
    1. Use the [Docugami API](https://api-docs.docugami.com), specifically the [documents](https://api-docs.docugami.com/#tag/documents/operation/upload-document) endpoint. You can also use the [docugami python library](https://pypi.org/project/docugami/) as a convenient wrapper.

Once your documents are in Docugami, they are processed and organized into sets of similar documents, e.g. NDAs, Lease Agreements, and Service Agreements. Docugami is not limited to any particular types of documents, and the clusters created depend on your particular documents. You can [change the docset assignments](https://help.docugami.com/home/working-with-the-doc-sets-view) later if you wish. You can monitor file status in the simple Docugami webapp, or use a [webhook](https://api-docs.docugami.com/#tag/webhooks) to be informed when your documents are done processing.

You can also use the [Docugami API](https://api-docs.docugami.com) or the  [docugami](https://pypi.org/project/docugami/) python library to do all the file processing without visiting the Docugami webapp except to get the API key.

> You can get an API key as documented here: https://help.docugami.com/home/docugami-api. This following code assumes you have set the `DOCUGAMI_API_TOKEN` environment variable.

First, let's define two simple helper methods to upload files and wait for them to finish processing.
"""
logger.info("## Data Loading")


DOCSET_NAME = "NTSB Aviation Incident Reports"
FILE_PATHS = [
    "/Users/tjaffri/ntsb/Report_CEN23LA277_192541.pdf",
    "/Users/tjaffri/ntsb/Report_CEN23LA338_192753.pdf",
    "/Users/tjaffri/ntsb/Report_CEN23LA363_192876.pdf",
    "/Users/tjaffri/ntsb/Report_CEN23LA394_192995.pdf",
    "/Users/tjaffri/ntsb/Report_ERA23LA114_106615.pdf",
    "/Users/tjaffri/ntsb/Report_WPR23LA254_192532.pdf",
]

assert len(FILE_PATHS) > 5, "Please provide at least 6 files"

dg_client = Docugami()
dg_docs = upload_to_named_docset(dg_client, FILE_PATHS, DOCSET_NAME)
dgml_paths = wait_for_dgml(dg_client, dg_docs)

plogger.debug(dgml_paths)

"""
If you are on the free Docugami tier, your files should be done in ~15 minutes or less depending on the number of pages uploaded and available resources (please contact Docugami for paid plans for faster processing). You can re-run the code above without reprocessing your files to continue waiting if your notebook is not continuously running (it does not re-upload).

### Partition PDF tables and text

You can use the [Docugami Loader](https://python.langchain.com/docs/integrations/document_loaders/docugami) to very easily get chunks for your documents, including semantic and structural metadata. This is the simpler and recommended approach for most use cases but in this notebook let's explore using the `dgml-utils` library to explore the segmented output for this file in more detail by processing the XML we just downloaded above.
"""
logger.info("### Partition PDF tables and text")


dgml_path = dgml_paths[Path(FILE_PATHS[0]).name]

with open(dgml_path, "r") as file:
    contents = file.read().encode("utf-8")

    chunks = get_chunks_str(
        contents,
        # Ensures Docugami XML semantic tags are included in the chunked output (set to False for text-only chunks and tables as Markdown)
        include_xml_tags=True,
        max_text_length=1024 * 8,  # 8k chars are ~2k tokens for Ollama.
    )

    logger.debug(f"found {len(chunks)} chunks, here are the first few")
    for chunk in chunks[:10]:
        logger.debug(chunk.text)

"""
The file processed by Docugami in the example above was [this one](https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain/GenerateNewestReport/192541/pdf) from the NTSB and you can look at the PDF side by side to compare the XML chunks above. 

If you want text based chunks instead, Docugami also supports those and renders tables as markdown:
"""
logger.info(
    "The file processed by Docugami in the example above was [this one](https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain/GenerateNewestReport/192541/pdf) from the NTSB and you can look at the PDF side by side to compare the XML chunks above.")

with open(dgml_path, "r") as file:
    contents = file.read().encode("utf-8")

    chunks = get_chunks_str(
        contents,
        include_xml_tags=False,  # text-only chunks and tables as Markdown
        max_text_length=1024
        * 8,  # 8k chars are ~2k tokens for Ollama. Ref: https://help.ollama.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    )

    logger.debug(f"found {len(chunks)} chunks, here are the first few")
    for chunk in chunks[:10]:
        logger.debug(chunk.text)

"""
## Docugami XML Deep Dive: Jane Doe NDA Example

Let's explore the Docugami XML output for a different example PDF file (a long form contract): [Jane Doe NDA](https://github.com/docugami/dgml-utils/blob/main/python/tests/test_data/article/Jane%20Doe%20NDA.pdf). We have provided processed Docugami XML output for this PDF here: https://github.com/docugami/dgml-utils/blob/main/python/tests/test_data/article/Jane%20Doe.xml so you can follow along without processing your own documents.
"""
logger.info("## Docugami XML Deep Dive: Jane Doe NDA Example")


dgml = requests.get(
    "https://raw.githubusercontent.com/docugami/dgml-utils/main/python/tests/test_data/article/Jane%20Doe.xml"
).text
chunks = get_chunks_str(dgml, include_xml_tags=True)
len(chunks)

category_counts = {}

for element in chunks:
    category = element.structure
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

category_counts

table_elements = [c for c in chunks if "table" in c.structure.split()]
logger.debug(f"There are {len(table_elements)} tables")

text_elements = [c for c in chunks if "table" not in c.structure.split()]
logger.debug(f"There are {len(text_elements)} text elements")

"""
The Docugami XML contains extremely detailed semantics and visual bounding boxes for all elements. The `dgml-utils` library parses text and non-text elements into formats appropriate to pass into LLMs (chunked text with XML semantic labels)
"""
logger.info("The Docugami XML contains extremely detailed semantics and visual bounding boxes for all elements. The `dgml-utils` library parses text and non-text elements into formats appropriate to pass into LLMs (chunked text with XML semantic labels)")

for element in text_elements[:20]:
    logger.debug(element.text)

logger.debug(table_elements[0].text)

"""
The XML markup contains structural as well as semantic tags, which provide additional semantics to the LLM for improved retrieval and generation.

If you prefer, you can set `include_xml_tags=False` in the `get_chunks_str` call above to not include XML markup. The text-only Docugami chunks are still very good since they follow the structural and semantic contours of the document rather than whitespace-only chunking. Tables are rendered as markdown in this case, so that some structural context is maintained even without the XML markup.
"""
logger.info("The XML markup contains structural as well as semantic tags, which provide additional semantics to the LLM for improved retrieval and generation.")

chunks_as_text = get_chunks_str(dgml, include_xml_tags=False)
table_elements_as_text = [
    c for c in chunks_as_text if "table" in c.structure.split()]

logger.debug(table_elements_as_text[0].text)

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

"""
### Add to vectorstore

Use [Multi Vector Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector#summary) with summaries: 

* `InMemoryStore` stores the raw text, tables
* `vectorstore` stores the embedded summaries
"""
logger.info("### Add to vectorstore")


def build_retriever(text_elements, tables, table_summaries):
    vectorstore = Chroma(
        collection_name="summaries", embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )

    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    texts = [i.text for i in text_elements]
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
    return retriever


retriever = build_retriever(text_elements, tables, table_summaries)

"""
## RAG

Run [RAG pipeline](https://python.langchain.com/docs/expression_language/cookbook/retrieval).
"""
logger.info("## RAG")


system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant that answers questions based on provided context. Your provided context can include text or tables, "
    "and may also contain semantic XML markup. Pay attention the semantic XML markup to understand more about the context semantics as "
    "well as structure (e.g. lists and tabular layouts expressed with HTML-like tags)"
)

human_prompt = HumanMessagePromptTemplate.from_template(
    """Context:

    {context}

    Question: {question}"""
)


def build_chain(retriever, model):
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    model = ChatOllama(model="llama3.2")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


chain = build_chain(retriever, model)

result = chain.invoke(
    "Name all the people authorized to receive confidential information, and their roles"
)
logger.debug(result)

"""
We can check the [trace](https://smith.langchain.com/public/21b3aa16-4ef3-40c3-92f6-3f0ceab2aedb/r) to see what chunks were retrieved.

This includes Table 1 in the doc, showing the disclosures table as XML markup (same one as above)

# RAG on Llama2 paper

Let's run the same Llama2 paper example from the [Semi_Structured_RAG.ipynb](./Semi_Structured_RAG.ipynb) notebook to see if we get the same results, and to contrast the table chunk returned by Docugami with the ones returned from Unstructured.
"""
logger.info("# RAG on Llama2 paper")

dgml = requests.get(
    "https://raw.githubusercontent.com/docugami/dgml-utils/main/python/tests/test_data/arxiv/2307.09288.xml"
).text
llama2_chunks = get_chunks_str(dgml, include_xml_tags=True)
len(llama2_chunks)

llama2_table_elements = [
    c for c in llama2_chunks if "table" in c.structure.split()]
logger.debug(f"There are {len(llama2_table_elements)} tables")

llama2_text_elements = [
    c for c in llama2_chunks if "table" not in c.structure.split()]
logger.debug(f"There are {len(llama2_text_elements)} text elements")

llama2_tables = [i.text for i in llama2_table_elements]
llama2_table_summaries = summarize_chain.batch(
    llama2_tables, {"max_concurrency": 5})

llama2_retriever = build_retriever(
    llama2_text_elements, llama2_tables, llama2_table_summaries
)

llama2_chain = build_chain(llama2_retriever, model)

llama2_chain.invoke("What is the number of training tokens for LLaMA2?")

"""
We can check the [trace](https://smith.langchain.com/public/5de100c3-bb40-4234-bf02-64bc708686a1/r) to see what chunks were retrieved.

This includes Table 1 in the doc, showing the tokens used for training table as semantic XML markup:

```xml
<table>
    <tbody>
        <tr>
            <td />
            <td>Training Data </td>
            <td>Params </td>
            <td>Context Length </td>
            <td>
                <Org>GQA </Org>
            </td>
            <td>Tokens </td>
            <td>LR </td>
        </tr>
        <tr>
            <td>Llama <Number>1 </Number></td>
            <td>
                <Llama1TrainingData>See <Person>Touvron </Person>et al. (<Number>2023</Number>) </Llama1TrainingData>
            </td>
            <td>
                <Llama1Params>
                    <Number>7B </Number>
                    <Number>13B </Number>
                    <Number>33B </Number>
                    <Number>65B </Number>
                </Llama1Params>
            </td>
            <td>
                <Llama1ContextLength>
                    <Number>2k </Number>
                    <Number>2k </Number>
                    <Number>2k </Number>
                    <Number>2k </Number>
                </Llama1ContextLength>
            </td>
            <td>
                <Llama1GQA>✗ ✗ ✗ ✗ </Llama1GQA>
            </td>
            <td>
                <Llama1Tokens><Number>1.0</Number>T <Number>1.0</Number>T <Number>1.4</Number>T <Number>
                    1.4</Number>T </Llama1Tokens>
            </td>
            <td>
                <Llama1LR> 3.0 × <Number>10−4 </Number> 3.0 × <Number>10−4 </Number> 1.5 × <Number>
                    10−4 </Number> 1.5 × <Number>10−4 </Number></Llama1LR>
            </td>
        </tr>
        <tr>
            <td>Llama <Number>2 </Number></td>
            <td>
                <Llama2TrainingData>A new mix of publicly available online data </Llama2TrainingData>
            </td>
            <td>
                <Llama2Params><Number>7B </Number>13B <Number>34B </Number><Number>70B </Number></Llama2Params>
            </td>
            <td>
                <Llama2ContextLength>
                    <Number>4k </Number>
                    <Number>4k </Number>
                    <Number>4k </Number>
                    <Number>4k </Number>
                </Llama2ContextLength>
            </td>
            <td>
                <Llama2GQA>✗ ✗ ✓ ✓ </Llama2GQA>
            </td>
            <td>
                <Llama2Tokens><Number>2.0</Number>T <Number>2.0</Number>T <Number>2.0</Number>T <Number>
                    2.0</Number>T </Llama2Tokens>
            </td>
            <td>
                <Llama2LR> 3.0 × <Number>10−4 </Number> 3.0 × <Number>10−4 </Number> 1.5 × <Number>
                    10−4 </Number> 1.5 × <Number>10−4 </Number></Llama2LR>
            </td>
        </tr>
    </tbody>
</table>
```

Finally, you can ask other questions that rely on more subtle parsing of the table, e.g.:
"""
logger.info(
    "We can check the [trace](https://smith.langchain.com/public/5de100c3-bb40-4234-bf02-64bc708686a1/r) to see what chunks were retrieved.")

llama2_chain.invoke("What was the learning rate for LLaMA2?")

"""
## Docugami KG-RAG Template

Docugami also provides a [langchain template](https://github.com/docugami/langchain-template-docugami-kg-rag) that you can integrate into your langchain projects.

Here's a walkthrough of how you can do this.

[![Docugami KG-RAG Walkthrough](https://img.youtube.com/vi/xOHOmL1NFMg/0.jpg)](https://www.youtube.com/watch?v=xOHOmL1NFMg)
"""
logger.info("## Docugami KG-RAG Template")

logger.info("\n\n[DONE]", bright=True)
