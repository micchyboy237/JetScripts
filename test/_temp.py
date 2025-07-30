import os
import shutil
from typing import List
from jet.code.markdown_utils._converters import convert_markdown_to_html
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType
from jet.scrapers.utils import extract_texts_by_hierarchy
from jet.wordnet.sentence import split_sentences
from jet.wordnet.text_chunker import chunk_texts
from shared.data_types.job import JobData
from jet.models.tokenizer.base import count_tokens


jobs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"

jobs: List[JobData] = load_file(jobs_file)

md_content = f"# Job Title: {jobs[0]["title"]}\n\n## Overview\n{jobs[0]["details"]}\nLink: {jobs[0]["link"]}"

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    embed_model: EmbedModelType = "mxbai-embed-large"

    sentences = split_sentences(md_content)
    token_counts: List[int] = count_tokens(
        embed_model, sentences, prevent_total=True)
    save_file([{"tokens": tokens, "sentence": sentence} for tokens, sentence in zip(
        token_counts, sentences)], f"{output_dir}/sentences.json")

    headers_with_heirarchy = derive_by_header_hierarchy(md_content)
    save_file(headers_with_heirarchy,
              f"{output_dir}/headers_with_heirarchy.json")

    results_ignore_links = base_parse_markdown(md_content, ignore_links=True)
    results_with_links = base_parse_markdown(md_content, ignore_links=False)

    save_file(results_ignore_links, f"{output_dir}/results_ignore_links.json")
    save_file(results_with_links, f"{output_dir}/results_with_links.json")

    html_str = convert_markdown_to_html(md_content)
    headings = extract_texts_by_hierarchy(html_str, ignore_links=True)
    save_file(headings, f"{output_dir}/headings.json")

    chunks = chunk_texts(md_content, chunk_size=64,
                         chunk_overlap=32, model=embed_model)
    token_counts: List[int] = count_tokens(
        embed_model, chunks, prevent_total=True)
    save_file([{"tokens": tokens, "chunk": chunk} for tokens, chunk in zip(
        token_counts, chunks)], f"{output_dir}/chunks.json")
