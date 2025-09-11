from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sagemaker_inference import encoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import json
import logging
import os
import shutil
import torch


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
# Cross Encoder Reranker

This notebook shows how to implement reranker in a retriever with your own cross encoder from [Hugging Face cross encoder models](https://huggingface.co/cross-encoder) or Hugging Face models that implements cross encoder function ([example: BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)). `SagemakerEndpointCrossEncoder` enables you to use these HuggingFace models loaded on Sagemaker.

This builds on top of ideas in the [ContextualCompressionRetriever](/docs/how_to/contextual_compression). Overall structure of this document came from [Cohere Reranker documentation](/docs/integrations/retrievers/cohere-reranker).

For more about why cross encoder can be used as reranking mechanism in conjunction with embeddings for better retrieval, refer to [Hugging Face Cross-Encoders documentation](https://www.sbert.net/examples/applications/cross-encoder/README.html).
"""
logger.info("# Cross Encoder Reranker")



def pretty_print_docs(docs):
    logger.debug(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

"""
## Set up the base vector store retriever
Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.
"""
logger.info("## Set up the base vector store retriever")


documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
embeddingsModel = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-distilbert-dot-v5"
)
retriever = FAISS.from_documents(texts, embeddingsModel).as_retriever(
    search_kwargs={"k": 20}
)

query = "What is the plan for the economy?"
docs = retriever.invoke(query)
pretty_print_docs(docs)

"""
## Doing reranking with CrossEncoderReranker
Now let's wrap our base retriever with a `ContextualCompressionRetriever`. `CrossEncoderReranker` uses `HuggingFaceCrossEncoder` to rerank the returned results.
"""
logger.info("## Doing reranking with CrossEncoderReranker")


model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("What is the plan for the economy?")
pretty_print_docs(compressed_docs)

"""
## Uploading Hugging Face model to SageMaker endpoint

Here is a sample `inference.py` for creating an endpoint that works with `SagemakerEndpointCrossEncoder`. For more details with step-by-step guidance, refer to [this article](https://huggingface.co/blog/kchoe/deploy-any-huggingface-model-to-sagemaker). 

It downloads Hugging Face model on the fly, so you do not need to keep the model artifacts such as `pytorch_model.bin` in your `model.tar.gz`.
"""
logger.info("## Uploading Hugging Face model to SageMaker endpoint")



PAIRS = "pairs"
SCORES = "scores"


class CrossEncoder:
    def __init__(self) -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info(f"Using device: {self.device}")
        model_name = "BAAI/bge-reranker-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)

    def __call__(self, pairs: List[List[str]]) -> List[float]:
        with torch.inference_mode():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = inputs.to(self.device)
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        return scores.detach().cpu().tolist()


def model_fn(model_dir: str) -> CrossEncoder:
    try:
        return CrossEncoder()
    except Exception:
        logging.exception(f"Failed to load model from: {model_dir}")
        raise


def transform_fn(
    cross_encoder: CrossEncoder, input_data: bytes, content_type: str, accept: str
) -> bytes:
    payload = json.loads(input_data)
    model_output = cross_encoder(**payload)
    output = {SCORES: model_output}
    return encoder.encode(output, accept)

logger.info("\n\n[DONE]", bright=True)