import sys
from typing import Generator, Optional, List
import os

from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
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
EVAL_MODEL = "gemma3:4b"

chunk_overlap = 40
chunk_size = 256

output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_llm_reranker"

query = "What are the steps in registering a National ID in the Philippines?"

# DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/llama_index/docs/docs/examples/data/10k/lyft_2021.pdf"
DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/valid-ids-scraper/philippines_national_id_registration_tips_2025/scraped_html.html"
CACHE_FILE = "llm_reranker.pkl"

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/Structured-LLMReranker-Lyft-10k.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Structured LLM Reranker Demonstration

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


class RelevantDocument(BaseModel):
    """Document that directly answers the query with specific, factual content."""

    document_number: int = Field(
        ...,
        description="Index of the document that includes direct, actionable, or factual information relevant to the query."
    )
    confidence: int = Field(
        ...,
        ge=1,
        le=10,
        description="Score from 1 to 10 showing the model's confidence in this document's relevance."
    )


class DocumentSelectionResult(BaseModel):
    """LLM response listing only documents with actual answers, and all that were evaluated."""

    relevant_documents: List[RelevantDocument] = Field(
        ...,
        description="List of documents with real, specific answers to the query. May be empty if none qualify."
    )
    evaluated_documents: List[int] = Field(
        ...,
        description="Document numbers that were evaluated for relevance by the model."
    )
    feedback: str = Field(
        ...,
        description="Overall explanation of which documents were selected and why, or why none qualified."
    )


output_cls = DocumentSelectionResult

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
all_nodes = [TextNode(text=h["content"], metadata={
                      "doc_index": idx}) for idx, h in enumerate(header_contents)]
# all_header_docs = [Document(text=h["content"], metadata={
#     "doc_index": idx}) for idx, h in enumerate(header_contents)]
# splitter = SentenceSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap,
#     tokenizer=get_words
# )
# all_nodes: list[TextNode] = splitter.get_nodes_from_documents(
#     documents=all_header_docs)
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

top_k = 5
top_node_texts = [
    node.text
    for node in nodes_with_scores[:top_k]
]
eval_result = evaluate_context_relevancy(
    EVAL_MODEL, query, top_node_texts)

if eval_result.passing:
    logger.success(
        f"Evaluation on context relevancy passed ({len(top_node_texts)})")
else:
    logger.error(
        f"Failed evaluation on context relevancy ({len(top_node_texts)})")
eval_context_relevancy_file = f"{output_dir}/eval_context_relevancy.json"
save_file(eval_result, eval_context_relevancy_file)

# Run LLM Chat
final_context = "\n\n".join(top_node_texts)
response = llm.chat(query, context=final_context)
llm_chat_history_file = f"{output_dir}/llm_chat_history.md"
chat_history_list = [
    f"## Query\n\n{query}",
    f"## Context\n\n{final_context}",
    f"## Response\n\n{response}",
]
chat_history = "\n\n".join(chat_history_list)
save_file(chat_history, llm_chat_history_file)


sys.exit()


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


def filter_relevant_documents(contexts: list[list[str]]) -> Generator[output_cls, None, None]:
    for context in contexts:
        prompt_tmpl = PromptTemplate(PROMPT_TEMPLATE)

        response = llm.structured_predict(
            output_cls=output_cls,
            prompt=prompt_tmpl,
            context="\n\n".join(context),
            query=query
        )

        yield response


def run_filter_relevant_documents(node_texts: list[str]) -> Generator[output_cls, None, None]:
    header_contents = merge_md_header_contents(
        [{"content": text} for text in node_texts], max_tokens=1500, tokenizer=get_ollama_tokenizer(LLM_MODEL).encode)
    header_docs = []
    for idx, header in tqdm(enumerate(header_contents), total=len(header_contents), desc="Chat..."):
        sub_headers = get_md_header_contents(header["content"])
        header_docs.append(
            [f"Document number: {node.metadata['doc_index'] + 1}\n{node.text}" for node in nodes_with_scores if node.text.splitlines()[0].strip() in [
                sub['content'].splitlines()[0].strip() for sub in sub_headers]]
        )

    results: list[output_cls] = []
    for response in tqdm(filter_relevant_documents(header_docs), total=len(header_contents)):
        results.append(response)
        yield response

    # Collect all relevant documents from the response
    if len(header_docs) > 1:
        # Extract texts from all relevant documents in all responses
        results_texts = []
        for response in results:
            for relevant_doc in response.relevant_documents:
                doc_index = relevant_doc.document_number - 1  # Convert to 0-based index
                results_texts.append(all_nodes[doc_index].text)

        # Now continue filtering with the new set of results_texts
        yield from run_filter_relevant_documents(results_texts)


if __name__ == "__main__":
    node_texts = [node.text for node in nodes_with_scores]

    all_results = []
    results_dict = {
        "query": query,
        "results": all_results
    }
    reranker_results_file = f"{output_dir}/llm_reranker_results.json"

    for response in run_filter_relevant_documents(node_texts):
        all_results.append(response)
        save_file(results_dict, reranker_results_file)

        results_texts = [
            all_nodes[relevant_doc.document_number - 1].text
            for response in all_results for relevant_doc in response.relevant_documents
        ]
        eval_result = evaluate_context_relevancy(
            EVAL_MODEL, query, results_texts)

        if eval_result.passing:
            logger.success(
                f"Evaluation on context relevancy passed ({len(all_results)})")
            break

    final_results = []
    seen_texts = set()  # To track unique texts
    for response in all_results:
        for relevant_doc in response.relevant_documents:
            doc_index = relevant_doc.document_number - 1  # Convert to 0-based index
            text = all_nodes[doc_index].text
            if text not in seen_texts:  # Ensure unique by text
                seen_texts.add(text)
                final_results.append(
                    {"confidence": relevant_doc.confidence, "text": text}
                )

    # Sort the final results by confidence in descending order
    final_results.sort(key=lambda x: x['confidence'], reverse=True)
    final_reranker_results_file = f"{output_dir}/final_llm_reranker_results.json"
    save_file(final_results, final_reranker_results_file)

    # Run LLM Chat
    final_context = "\n\n".join(item["text"] for item in final_results)
    response = llm.chat(query, context=final_context)
    llm_chat_history_file = f"{output_dir}/llm_chat_history.md"
    chat_history_list = [
        f"## Query\n\n{query}",
        f"## Context\n\n{final_context}",
        f"## Response\n\n{response}",
    ]
    chat_history = "\n\n".join(chat_history_list)
    save_file(chat_history, llm_chat_history_file)
