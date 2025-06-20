import os
from typing import List, TypedDict, Optional
from jet.file.utils import load_file, save_file
from jet.llm.mlx.mlx_types import LLMModelType
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import get_tokenizer
from jet.vectors.document_types import HeaderDocument
from jet.logger import logger
import tiktoken
from sentence_transformers import SentenceTransformer, util
import numpy as np

from jet.wordnet.similarity import GroupedResult
from jet.wordnet.text_chunker import chunk_headers


class MergedDocument(TypedDict):
    """Represents a merged document from a group."""
    text: str
    headers: List[str]
    source_urls: List[str]
    doc_indexes: List[int]
    token_count: int


def merge_grouped_documents(
    group: GroupedResult,
    max_tokens: int,
    query: Optional[str] = None,
    model_name: EmbedModelType = "static-retrieval-mrl-en-v1",
) -> MergedDocument:
    """
    Merges a group of similar documents into a single document that does not exceed max_tokens.
    """
    try:
        tokenizer = get_tokenizer(model_name)
        model = SentenceTransformer(model_name, device="cpu", backend="onnx")
    except Exception as e:
        logger.log("merge_grouped_documents:",
                   f"Initialization failed: {str(e)}", colors=["WHITE", "RED"])
        raise

    headers = group["headers"]
    contents = group["contents"]
    source_urls = group["source_urls"]
    doc_indexes = group["doc_indexes"]

    if not contents:
        logger.log("merge_grouped_documents:",
                   "Empty group provided", colors=["WHITE", "YELLOW"])
        return {
            "text": "",
            "headers": [],
            "source_urls": [],
            "doc_indexes": [],
            "token_count": 0
        }

    try:
        content_embeddings = model.encode(contents, convert_to_tensor=True)
        unique_indices = []
        seen = set()
        for i, emb in enumerate(content_embeddings):
            if i not in seen:
                unique_indices.append(i)
                similarities = util.cos_sim(emb, content_embeddings).flatten()
                for j, sim in enumerate(similarities):
                    if sim > 0.95 and i != j:
                        seen.add(j)
        unique_contents = [contents[i] for i in unique_indices]
        unique_headers = [headers[i] for i in unique_indices]
        unique_source_urls = [source_urls[i] for i in unique_indices]
        unique_doc_indexes = [doc_indexes[i] for i in unique_indices]
        logger.log("merge_grouped_documents:",
                   f"Deduplicated to {len(unique_contents)} unique documents", colors=["WHITE", "BLUE"])
    except Exception as e:
        logger.log("merge_grouped_documents:",
                   f"Deduplication failed: {str(e)}", colors=["WHITE", "RED"])
        raise

    sentences = []
    sentence_sources = []
    for i, content in enumerate(unique_contents):
        content_sentences = [s.strip()
                             for s in content.split(".") if s.strip()]
        sentences.extend(content_sentences)
        sentence_sources.extend(
            [(i, unique_headers[i], unique_source_urls[i], unique_doc_indexes[i])] * len(content_sentences))

    try:
        if sentences:
            sentence_embeddings = model.encode(
                sentences, convert_to_tensor=True)
            selected_indices = []
            seen_embeddings = []
            for i, emb in enumerate(sentence_embeddings):
                if not seen_embeddings or all(
                    util.cos_sim(emb, np.array(seen_emb)).flatten().max() < 0.9
                    for seen_emb in seen_embeddings
                ):
                    selected_indices.append(i)
                    seen_embeddings.append(emb)

            selected_sentences = [sentences[i] for i in selected_indices]
            selected_sources = [sentence_sources[i] for i in selected_indices]
        else:
            selected_sentences = []
            selected_sources = []

        logger.log("merge_grouped_documents:",
                   f"Selected {len(selected_sentences)} diverse sentences", colors=["WHITE", "BLUE"])
    except Exception as e:
        logger.log("merge_grouped_documents:",
                   f"Failed to select diverse sentences: {str(e)}", colors=["WHITE", "RED"])
        raise

    try:
        header_text = [f"# {h}" for h in set(unique_headers)]
        merged_text = "\n".join(header_text)
        current_tokens = len(tokenizer.encode(merged_text))

        content_text = []
        used_source_indices = set()
        for i, sentence in enumerate(selected_sentences):
            sentence_tokens = len(tokenizer.encode(sentence))
            if current_tokens + sentence_tokens <= max_tokens * 0.8:
                content_text.append(sentence)
                used_source_indices.add(selected_sources[i][0])
                current_tokens += sentence_tokens

        if content_text:
            merged_text += "\n\n" + ". ".join(content_text) + "."

        used_urls = [unique_source_urls[i] for i in used_source_indices]
        url_text = "\n\nSources:\n" + \
            "\n".join([f"- {url}" for url in set(used_urls)])
        url_tokens = len(tokenizer.encode(url_text))

        if current_tokens + url_tokens <= max_tokens:
            merged_text += url_text
            current_tokens += url_tokens
        else:
            truncated_urls = []
            for url in set(used_urls):
                temp_text = merged_text + "\n- " + url
                if len(tokenizer.encode(temp_text)) <= max_tokens:
                    truncated_urls.append(url)
            if truncated_urls:
                merged_text += "\n\nSources:\n" + \
                    "\n".join([f"- {url}" for url in truncated_urls])
            current_tokens = len(tokenizer.encode(merged_text))

        logger.log("merge_grouped_documents:",
                   f"Merged document created with {current_tokens} tokens", colors=["WHITE", "GREEN"])
    except Exception as e:
        logger.log("merge_grouped_documents:",
                   f"Merging failed: {str(e)}", colors=["WHITE", "RED"])
        raise

    return {
        "text": merged_text,
        "headers": list(set(unique_headers)),
        "source_urls": list(set(unique_source_urls)),
        "doc_indexes": list(set([unique_doc_indexes[i] for i in used_source_indices])),
        "token_count": current_tokens
    }


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    docs = load_file(docs_file)
    query = docs["query"]
    docs = docs["documents"]
    docs = [HeaderDocument(**doc) for doc in docs]
    chunked_docs = chunk_headers(docs, max_tokens=200, model=llm_model)
    docs = chunked_docs

    # Sample mock data
    sample_group: GroupedResult = {
        "headers": [
            "Climate Change Effects",
            "Climate Change Effects",
            "Global Warming Impact"
        ],
        "contents": [
            "Climate change is causing extreme weather events. Rising temperatures disrupt ecosystems.",
            "Rising temperatures disrupt ecosystems. Climate change is linked to wildfires.",
            "Climate change is causing extreme weather events and rising sea levels."
        ],
        "source_urls": [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3"
        ],
        "doc_indexes": [0, 1, 2]
    }

    # Configuration
    max_tokens = 500
    embed_model: EmbedModelType = "static-retrieval-mrl-en-v1"

    try:
        merged = merge_grouped_documents(
            group=sample_group,
            max_tokens=max_tokens,
            query="climate change effects",
            model_name=embed_model,
        )

        print("----- Merged Document Output -----\n")
        print("Merged Text:\n", merged["text"])
        print("\nHeaders:", merged["headers"])
        print("Sources:", merged["source_urls"])
        print("Doc Indexes:", merged["doc_indexes"])
        print("Token Count:", merged["token_count"])
        save_file(merged, f"{output_dir}/results.json")

    except Exception as e:
        logger.log("main", f"Error during merge: {str(e)}", colors=[
                   "WHITE", "RED"])
