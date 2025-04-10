from llama_index.core import PromptTemplate
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


class Answer(BaseModel):
    title: str = Field(
        ..., description="The exact title of the anime, as it appears in the document.")
    document_number: int = Field(
        ..., description="The number of the document that includes this anime (e.g., 'Document number: 3').")
    release_year: Optional[int] = Field(
        description="The most recent known release year of the anime, if specified in the document.")


class QueryResponse(BaseModel):
    results: list[Answer] = Field(
        default_factory=list,
        description="List of relevant anime titles extracted from the documents, matching the user's query. Each entry includes the title, source document number, and release year (if known)."
    )


output_cls = QueryResponse


instruction = """
Extract relevant information from the documents that directly answer the query.

- Use only the content from the documents provided.
- Remove duplicates when found.
- Return only the generated JSON value without any explanations surrounded by ```json that adheres to the model below:

Schema:
{schema_str}

Example output:
```json
[
    {{
        "title": "Anime Title 1",
        "document_number": 2,
        "release_year": 2020
    }},
    {{
        "title": "Anime Title 2",
        "document_number": 3,
        "release_year": 2023
    }}
]
""".strip()
instruction = instruction.format(schema_str=class_to_string(output_cls))

prompt_template = PromptTemplate("""
--- Documents ---
{headers}
--- End of Documents ---

Instructions:
You are given a set of structured documents. Your task is to extract all answers relevant to the query using only the content within the documents.

- Use the schema shown below to return your result.
- Only return answers found directly in the documents.
- Remove any duplicates.
- Return *only* the final JSON enclosed in a ```json block.

Schema:
{schema}

Query:
{query}

Answer:
""")


def strip_left_hashes(text: str) -> str:
    """
    Removes all leading '#' characters from lines that start with '#'.
    Also strips surrounding whitespace from those lines.

    Args:
        text (str): The input multiline string

    Returns:
        str: Modified string with '#' and extra whitespace removed from matching lines
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            cleaned_lines.append(stripped_line.lstrip('#').strip())
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


if __name__ == "__main__":
    query = "Top otome villainess anime today"

    # llm_model = "gemma3:4b"
    llm_model = "mistral"
    embed_models = [
        "paraphrase-multilingual",
        # "mxbai-embed-large",
    ]
    embed_model = embed_models[0]
    sub_chunk_size = 128
    sub_chunk_overlap = 40

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
        reranked_sub_text = strip_left_hashes(reranked_sub_text)
        header_texts.append(
            f"Document number: {doc.metadata["doc_index"] + 1}\n```text\n{doc.metadata["header"]}\n{reranked_sub_text}\n```"
        )

    header_texts = filter_texts(header_texts, llm_model)

    headers = "\n\n".join(header_texts)
    save_file(headers, os.path.join(
        output_dir, "filter_header_docs/top_nodes.md"))

    llm = Ollama(temperature=0.3, model=llm_model)

    # message = prompt_template.format(
    #     headers=headers,
    #     query=query,
    #     instruction=instruction,
    #     schema=class_to_string(output_cls),
    # )

    response = llm.chat(
        prompt_template,
        model=llm_model,
        template_vars={
            "headers": headers,
            "instruction": instruction,
            "schema": output_cls.model_json_schema(),
            "query": query,
        }
    )
    json_result = extract_json_block_content(str(response))
    results = json.loads(json_result)

    logger.success(format_json(results))

    save_file({
        "query": query,
        "results": results
    }, f"{output_dir}/results.json")
