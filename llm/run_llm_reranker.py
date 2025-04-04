import os

from llama_index.core.prompts.base import PromptTemplate
from tqdm import tqdm
from jet.cache.joblib.utils import load_persistent_cache, save_persistent_cache, ttl_cache
from jet.code.splitter_markdown_utils import extract_md_header_contents, get_md_header_contents, merge_md_header_contents
from jet.file.utils import load_file, save_file
from jet.scrapers.preprocessor import html_to_markdown
from jet.vectors.reranker.bm25_helpers import HybridSearch
from jet.wordnet.similarity import get_query_similarity_scores
from jet.wordnet.words import get_words
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from pydantic import BaseModel, Field
from jet.logger import logger
from jet.token.token_utils import get_model_max_tokens, get_ollama_tokenizer, token_counter
from jet.utils.commands import copy_to_clipboard
from llama_index.core import SimpleDirectoryReader
from jet.vectors.reranker.helpers.structured_llm_rerank import StructuredLLMRerank
from jet.llm.ollama.base import Ollama, OllamaEmbedding, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
import pandas as pd
from copy import deepcopy

from llama_index.core.schema import Document, NodeWithScore, TextNode

# LLM_MODEL = "llama3.2"
LLM_MODEL = "gemma3:4b"
LLM_MAX_TOKENS = get_model_max_tokens(LLM_MODEL)
EMBED_MODEL = "mxbai-embed-large"
EMBED_MAX_TOKENS = get_model_max_tokens(EMBED_MODEL)

chunk_overlap = 40
chunk_size = 256

output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_llm_reranker"

query = "What are the steps in registering a National ID in the Philippines?"

# DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/llama_index/docs/docs/examples/data/10k/lyft_2021.pdf"
DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/valid-ids-scraper/philippines_national_id_registration_tips_2025/scraped_html.html"
CACHE_FILE = "llm_reranker.pkl"

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/Structured-LLMReranker-Lyft-10k.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Structured LLM Reranker Demonstration (2021 Lyft 10-k)

This tutorial showcases how to do a two-stage pass for retrieval. Use embedding-based retrieval with a high top-k value
in order to maximize recall and get a large set of candidate items. Then, use LLM-based retrieval
to dynamically select the nodes that are actually relevant to the query using structured output.

Usage of `StructuredLLMReranker` is preferred over `LLMReranker` when you are using a model that supports function calling.
This class will make use of the structured output capability of the model instead of relying on prompting the model to rank the nodes in a desired format.
"""

# %pip install llama-index-llms-ollama

# import nest_asyncio

# nest_asyncio.apply()


"""
## Download Data
"""

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

"""
## Custom output cls structure
"""


class DocumentWithRelevance(BaseModel):
    """Document rankings as selected by model."""

    document_number: int = Field(
        description="The number of the document within the provided list"
    )
    relevance: int = Field(
        description="Relevance score from 1-10 of the document to the given query, based on the document content. The document must contain information that directly answers or provides substantial evidence for the query to be considered relevant.",
        json_schema_extra={"minimum": 1, "maximum": 10},
    )


class DocumentRelevanceList(BaseModel):
    """List of documents with relevance scores."""

    documents: list[DocumentWithRelevance] = Field(
        description="List of documents with relevance scores"
    )
    feedback: str = Field(
        description="Overall feedback on the relevance of documents.",
    )


document_relevance_list_cls = DocumentRelevanceList

"""
## Load Data, Build Index
"""


llm = Ollama(temperature=0.3, model=LLM_MODEL,
             request_timeout=300.0, context_window=LLM_MAX_TOKENS)
embed_model = OllamaEmbedding(model_name=EMBED_MODEL)


html: str = load_file(DATA_FILE)

# documents = [doc for doc in documents if "covid" in doc.text.lower()]


# splitter = SentenceSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap,
#     tokenizer=get_words
# )
# all_nodes = splitter.get_nodes_from_documents(documents=documents)

md_text = html_to_markdown(html)
header_contents = get_md_header_contents(md_text)
# header_contents = extract_md_header_contents(
#             md_text, min_tokens_per_chunk=256, max_tokens_per_chunk=int(chat_max_tokens * 0.65), tokenizer=get_ollama_tokenizer(embed_model).encode)
all_nodes = [TextNode(text=h["content"], metadata={
                      "doc_index": idx}) for idx, h in enumerate(header_contents)]
all_texts = [node.text for node in all_nodes]
all_texts_dict = {node.text: node for node in all_nodes}


query_similarities = get_query_similarity_scores(
    query, all_texts, model_name=EMBED_MODEL)
nodes_with_scores = [
    NodeWithScore(
        node=TextNode(text=text,
                      metadata=all_texts_dict[text].metadata),
        score=score
    )
    for text, score in query_similarities[0]["results"].items()
]
nodes_with_scores_file = f"{output_dir}/nodes_with_scores.json"
save_file({
    "query": query,
    "results": nodes_with_scores
}, nodes_with_scores_file)

header_contents = merge_md_header_contents(
    [{"content": node.text} for node in nodes_with_scores], max_tokens=1500, tokenizer=get_ollama_tokenizer(LLM_MODEL).encode)
header_texts = []
for idx, header in tqdm(enumerate(header_contents), total=len(header_contents), desc="Chat..."):
    sub_headers = get_md_header_contents(header["content"])
    header_texts.append(
        [f"Document number: {node.metadata['doc_index'] + 1}\n{node.text}" for node in nodes_with_scores if node.text.splitlines()[0].strip() in [
            sub['content'].splitlines()[0].strip() for sub in sub_headers]]
    )


merged_token_counts = token_counter(
    header_texts, LLM_MODEL, prevent_total=True)

PROMPT_TEMPLATE = """
A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided. 
Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well 
as the relevance score. The relevance score is a number from 1-10 based on how relevant you think the document is to the question.
Do not include any documents that are not relevant to the question. 
Let's try this now: 

{context}

Query: {query}
Answer:
""".strip()

results = []
results_dict = {
    "query": query,
    "results": results
}
reranker_results_file = f"{output_dir}/structured_llm_reranker_results.json"
for context in header_texts:
    prompt_tmpl = PromptTemplate(PROMPT_TEMPLATE)

    response = llm.structured_predict(
        output_cls=document_relevance_list_cls,
        prompt=prompt_tmpl,
        context=context,
        query=query
    )
    results.append(response)
    save_file(results_dict, reranker_results_file)
