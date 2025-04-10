from jet.utils.class_utils import class_to_string
from pydantic import BaseModel, Field
import json
import os
from typing import Optional

from jet.features.scrape_search_chat import get_docs_from_html, get_nodes_from_docs, get_nodes_parent_mapping, rerank_nodes
from jet.file.utils import load_file, save_file
from jet.llm.ollama.base import Ollama, OllamaEmbedding
from jet.logger import logger
from jet.token.token_utils import filter_texts, split_docs
from jet.transformers.formatters import format_json
from jet.utils.markdown import extract_json_block_content
from jet.wordnet.sentence import split_sentences
from tqdm import tqdm


prompt_template = """
--- Documents ---
{headers}
--- End of Documents ---

Instruction:
{instruction}
Query: {query}
Answer:
""".strip()


class Answer(BaseModel):
    anime_title: str = Field(
        ..., description="The anime title as it appears in the provided document.")
    document: int = Field(
        ..., description="The document number as indicated in the formatted input (e.g., 'Document number').")
    year: Optional[int] = Field(
        description="Latest release year of the anime, if available.")


class QueryResponse(BaseModel):
    results: list[Answer] = Field(
        [],
        description="A list of answers extracted only from documents that contain relevant information matching the query."
    )


output_cls = QueryResponse


if __name__ == "__main__":
    # llm_model = "gemma3:4b"
    llm_model = "mistral"
    embed_models = [
        "paraphrase-multilingual",
        # "mxbai-embed-large",
    ]
    embed_model = embed_models[0]

    sub_chunk_size = 128
    sub_chunk_overlap = 40

    instruction = "Given the provided documents, select all that contains answers to the query in JSON format.\n\nDeduplicate answers if possible\nReturn only the generated JSON value without any explanations surrounded by ```json that adheres to the model below:\n{schema_str}```"
    instruction.format(schema_str=class_to_string(output_cls))
    query = "Top otome villainess anime today"

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_anime_scraper/myanimelist_net/scraped_html.html"
    output_dir = os.path.join(os.path.dirname(__file__), "generated")

    html: str = load_file(html_file)
    header_docs = get_docs_from_html(html)
    embed_model = OllamaEmbedding(model_name=embed_model)
    header_tokens: list[list[int]] = embed_model.encode(
        [d.text for d in header_docs])

    header_texts = []
    for doc_idx, doc in tqdm(enumerate(header_docs), total=len(header_docs)):
        sub_nodes = split_docs(
            doc, llm_model, tokens=header_tokens[doc_idx], chunk_size=sub_chunk_size, chunk_overlap=sub_chunk_overlap)
        parent_map = get_nodes_parent_mapping(sub_nodes, header_docs)

        sub_query = f"Query: {query}\n{doc.metadata["header"]}"
        reranked_sub_nodes = rerank_nodes(
            sub_query, sub_nodes, embed_models, parent_map)

        reranked_sub_text = "\n".join([n.text for n in reranked_sub_nodes[:3]])
        reranked_sub_text = reranked_sub_text.lstrip(
            doc.metadata["header"]).strip()
        header_texts.append(
            f"Document number: {doc.metadata["doc_index"] + 1}\n```text\n{doc.metadata["header"].lstrip("#").strip()}\n{reranked_sub_text}\n```"
        )

    header_texts = filter_texts(header_texts, llm_model)

    headers = "\n\n".join(header_texts)
    save_file(headers, os.path.join(
        output_dir, "filter_header_docs/top_nodes.md"))

    llm = Ollama(temperature=0.0, model=llm_model)

    message = prompt_template.format(
        headers=headers,
        query=query,
        instruction=instruction,
    )

    response = llm.chat(message, model=llm_model)
    json_result = extract_json_block_content(str(response))
    result = json.loads(json_result)

    logger.success(format_json(result))
