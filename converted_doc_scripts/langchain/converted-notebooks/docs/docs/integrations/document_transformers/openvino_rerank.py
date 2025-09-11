from jet.logger import logger
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenVINOEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
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
# OpenVINO Reranker

[OpenVINO™](https://github.com/openvinotoolkit/openvino) is an open-source toolkit for optimizing and deploying AI inference. The OpenVINO™ Runtime supports various hardware [devices](https://github.com/openvinotoolkit/openvino?tab=readme-ov-file#supported-hardware-matrix) including x86 and ARM CPUs, and Intel GPUs. It can help to boost deep learning performance in Computer Vision, Automatic Speech Recognition, Natural Language Processing and other common tasks.

Hugging Face rerank model can be supported by OpenVINO through ``OpenVINOReranker`` class. If you have an Intel GPU, you can specify `model_kwargs={"device": "GPU"}` to run inference on it.
"""
logger.info("# OpenVINO Reranker")

# %pip install --upgrade-strategy eager "optimum[openvino,nncf]" --quiet
# %pip install --upgrade --quiet  faiss-cpu

def pretty_print_docs(docs):
    logger.debug(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i + 1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

"""
## Set up the base vector store retriever
Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.
"""
logger.info("## Set up the base vector store retriever")


documents = TextLoader(
    "../../how_to/state_of_the_union.txt",
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = OpenVINOEmbeddings(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2"
)
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)

"""
## Reranking with OpenVINO
Now let's wrap our base retriever with a `ContextualCompressionRetriever`, using `OpenVINOReranker` as a compressor.
"""
logger.info("## Reranking with OpenVINO")


model_name = "BAAI/bge-reranker-large"

ov_compressor = OpenVINOReranker(model_name_or_path=model_name, top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=ov_compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
logger.debug([doc.metadata["id"] for doc in compressed_docs])

"""
After reranking, the top 4 documents are different from the top 4 documents retrieved by the base retriever.
"""
logger.info("After reranking, the top 4 documents are different from the top 4 documents retrieved by the base retriever.")

pretty_print_docs(compressed_docs)

"""
## Export IR model
It is possible to export your rerank model to the OpenVINO IR format with ``OVModelForSequenceClassification``, and load the model from local folder.
"""
logger.info("## Export IR model")


ov_model_dir = "bge-reranker-large-ov"
if not Path(ov_model_dir).exists():
    ov_compressor.save_model(ov_model_dir)

ov_compressor = OpenVINOReranker(model_name_or_path=ov_model_dir)

"""
For more information refer to:

* [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).

* [OpenVINO Documentation](https://docs.openvino.ai/2024/home.html).

* [OpenVINO Get Started Guide](https://www.intel.com/content/www/us/en/content-details/819067/openvino-get-started-guide.html).

* [RAG Notebook with LangChain](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-rag-langchain).
"""
logger.info("For more information refer to:")

logger.info("\n\n[DONE]", bright=True)