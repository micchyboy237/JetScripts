from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.evaluation.benchmarks import HotpotQAEvaluator
from llama_index.core.postprocessor import SentenceTransformerRerank
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/HotpotQADistractor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# HotpotQADistractor Demo

This notebook walks through evaluating a query engine using the HotpotQA dataset. In this task, the LLM must answer a question given a pre-configured context. The answer usually has to be concise, and accuracy is measured by calculating the overlap (measured by F1) and exact match.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# HotpotQADistractor Demo")

# %pip install llama-index-llms-ollama

# !pip install llama-index


llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
embed_model = resolve_embed_model(
    "local:sentence-transformers/all-MiniLM-L6-v2"
)

index = VectorStoreIndex.from_documents(
    [Document.example()], embed_model=embed_model, show_progress=True
)

"""
First we try with a very simple engine. In this particular benchmark, the retriever and hence index is actually ignored, as the documents retrieved for each query is provided in the dataset. This is known as the "distractor" setting in HotpotQA.
"""
logger.info("First we try with a very simple engine. In this particular benchmark, the retriever and hence index is actually ignored, as the documents retrieved for each query is provided in the dataset. This is known as the "distractor" setting in HotpotQA.")

engine = index.as_query_engine(llm=llm)

HotpotQAEvaluator().run(engine, queries=5, show_result=True)

"""
Now we try with a sentence transformer reranker, which selects 3 out of the 10 nodes proposed by the retriever
"""
logger.info("Now we try with a sentence transformer reranker, which selects 3 out of the 10 nodes proposed by the retriever")


rerank = SentenceTransformerRerank(top_n=3)

engine = index.as_query_engine(
    llm=llm,
    node_postprocessors=[rerank],
)

HotpotQAEvaluator().run(engine, queries=5, show_result=True)

"""
The F1 and exact match scores appear to improve slightly.

Note that the benchmark optimizes for producing short factoid answers without explanations, although it is known that CoT prompting can sometimes help in output quality. 

The scores used are also not a perfect measure of correctness, but can be a quick way to identify how changes in your query engine change the output.
"""
logger.info("The F1 and exact match scores appear to improve slightly.")

logger.info("\n\n[DONE]", bright=True)