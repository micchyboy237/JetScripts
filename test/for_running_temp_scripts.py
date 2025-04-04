import os
from typing import List, Dict, Optional

from pydantic import BaseModel, Field
from jet.logger import logger
from jet.file.utils import load_file, save_file
from jet.scrapers.preprocessor import html_to_markdown
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.token.token_utils import get_model_max_tokens, split_docs
from jet.wordnet.similarity import get_query_similarity_scores
from llama_index.core.schema import Document, NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.ollama.base import Ollama, OllamaEmbedding

# --- Constants ---
LLM_MODEL = "gemma3:4b"
EMBED_MODELS = [
    "mxbai-embed-large",
    "paraphrase-multilingual",
    "granite-embedding",
]
EVAL_MODEL = LLM_MODEL
DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/valid-ids-scraper/philippines_national_id_registration_tips_2025/scraped_html.html"
OUTPUT_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/run_llm_reranker"
QUERY = "What are the steps in registering a National ID in the Philippines?"
TOP_K = 5

# --- Pydantic Classes ---


class RelevantDocument(BaseModel):
    document_number: int = Field(..., ge=0)
    confidence: int = Field(..., ge=1, le=10)


class DocumentSelectionResult(BaseModel):
    relevant_documents: List[RelevantDocument]
    evaluated_documents: List[int]
    feedback: str

# --- Core Functions ---


def get_docs_from_html(html: str) -> list[Document]:
    md_text = html_to_markdown(html)
    header_contents = get_md_header_contents(md_text)
    docs = [
        Document(
            text=header["content"],
            metadata={
                "doc_index": i,
                "header": header["header"],
                "header_level": header["header_level"],
            }
        )
        for i, header in enumerate(header_contents)
    ]
    return docs


def get_nodes_from_docs(docs: list[Document], chunk_size: Optional[int] = None, chunk_overlap: int = 40) -> tuple[list[TextNode], dict[str, TextNode]]:
    model = min(EMBED_MODELS, key=get_model_max_tokens)
    chunk_size = chunk_size or get_model_max_tokens(model)

    nodes = split_docs(docs, model=model, chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap)
    parent_map = {}
    for node in nodes:
        if node.parent_node and not node.parent_node.node_id in parent_map:
            parent_doc = docs[node.parent_node.metadata["doc_index"]]
            parent_node = TextNode(
                node_id=node.parent_node.node_id,
                text=parent_doc.text,
                metadata=node.parent_node.metadata
            )
            parent_map[node.parent_node.node_id] = parent_node

    return nodes, parent_map


def rerank_nodes(query: str, nodes: List[TextNode], embed_models: List[str], parent_map: Dict[str, TextNode]) -> List[NodeWithScore]:
    texts = [n.text for n in nodes]
    node_map = {n.text: n for n in nodes}
    query_scores = get_query_similarity_scores(
        query, texts, model_name=embed_models)

    results = []
    seen_docs = set()
    for text, score in query_scores[0]["results"].items():
        node = node_map[text]
        parent_info = node.relationships.get(NodeRelationship.PARENT)

        parent_text = text  # fallback to child text
        if parent_info and isinstance(parent_info, RelatedNodeInfo):
            parent_node = parent_map.get(parent_info.node_id)
            if parent_node:
                parent_text = parent_node.text

        doc_index = node.metadata["doc_index"]
        if doc_index not in seen_docs:
            seen_docs.add(doc_index)
            results.append(NodeWithScore(node=TextNode(
                text=parent_text, metadata=node.metadata), score=score))

    return results


def evaluate_relevancy_and_save(query: str, top_nodes: List[NodeWithScore], output_dir: str) -> bool:
    eval_result = evaluate_context_relevancy(
        EVAL_MODEL, query, [n.text for n in top_nodes])
    save_file(eval_result, os.path.join(
        output_dir, "eval_context_relevancy.json"))
    if eval_result.passing:
        logger.success(f"Context relevancy passed ({len(top_nodes)})")
    else:
        logger.error(f"Context relevancy failed ({len(top_nodes)})")
    return eval_result.passing


def run_llm_chat(query: str, top_nodes: List[NodeWithScore], output_dir: str, llm) -> str:
    context = "\n\n".join([n.text for n in top_nodes])
    response = llm.chat(query, context=context)
    history = "\n\n".join([
        f"## Query\n\n{query}",
        f"## Context\n\n{context}",
        f"## Response\n\n{response}",
    ])
    save_file(history, os.path.join(output_dir, "llm_chat_history.md"))
    return response

# --- Main ---


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    html = load_file(DATA_FILE)

    # Setup nodes
    docs = get_docs_from_html(html)
    nodes, parent_map = get_nodes_from_docs(docs)

    # Search nodes
    reranked_nodes = rerank_nodes(QUERY, nodes, EMBED_MODELS, parent_map)
    save_file({
        "query": QUERY,
        "results": reranked_nodes
    }, os.path.join(OUTPUT_DIR, "reranked_nodes.json"))

    top_nodes = reranked_nodes[:TOP_K]

    # Evaluate contexts
    eval_result = evaluate_context_relevancy(
        EVAL_MODEL, QUERY, [n.text for n in top_nodes])
    save_file(eval_result, os.path.join(
        OUTPUT_DIR, "eval_context_relevancy.json"))
    if eval_result.passing:
        logger.success(f"Context relevancy passed ({len(top_nodes)})")
    else:
        logger.error(f"Context relevancy failed ({len(top_nodes)})")
        return

    # Chat LLM
    llm = Ollama(temperature=0.3, model=LLM_MODEL,
                 request_timeout=300.0, context_window=get_model_max_tokens(LLM_MODEL))
    context = "\n\n".join([n.text for n in top_nodes])
    response = llm.chat(QUERY, context=context)
    history = "\n\n".join([
        f"## Query\n\n{QUERY}",
        f"## Context\n\n{context}",
        f"## Response\n\n{response}",
    ])
    save_file(history, os.path.join(OUTPUT_DIR, "llm_chat_history.md"))


if __name__ == "__main__":
    main()
