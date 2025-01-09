from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.embeddings import resolve_embed_model
from jet.llm.ollama.base import Ollama
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation.benchmarks import HotpotQAEvaluator
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/HotpotQADistractor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# HotpotQADistractor Demo
#
# This notebook walks through evaluating a query engine using the HotpotQA dataset. In this task, the LLM must answer a question given a pre-configured context. The answer usually has to be concise, and accuracy is measured by calculating the overlap (measured by F1) and exact match.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-llms-ollama

# !pip install llama-index


llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
embed_model = resolve_embed_model(
    "local:sentence-transformers/all-MiniLM-L6-v2"
)

index = VectorStoreIndex.from_documents(
    [Document.example()], embed_model=embed_model, show_progress=True
)

# First we try with a very simple engine. In this particular benchmark, the retriever and hence index is actually ignored, as the documents retrieved for each query is provided in the dataset. This is known as the "distractor" setting in HotpotQA.

engine = index.as_query_engine(llm=llm)

HotpotQAEvaluator().run(engine, queries=5, show_result=True)

# Now we try with a sentence transformer reranker, which selects 3 out of the 10 nodes proposed by the retriever


rerank = SentenceTransformerRerank(top_n=3)

engine = index.as_query_engine(
    llm=llm,
    node_postprocessors=[rerank],
)

HotpotQAEvaluator().run(engine, queries=5, show_result=True)

# The F1 and exact match scores appear to improve slightly.
#
# Note that the benchmark optimizes for producing short factoid answers without explanations, although it is known that CoT prompting can sometimes help in output quality.
#
# The scores used are also not a perfect measure of correctness, but can be a quick way to identify how changes in your query engine change the output.

logger.info("\n\n[DONE]", bright=True)
