import math
import os
import re
import shutil
from typing import Dict, List
from jet.code.splitter_markdown_utils import get_md_header_docs
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.tasks.answer_multiple_choice_with_key import answer_multiple_choice_with_key, AnswerResult
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.utils.search_docs import search_docs
from jet.logger import logger
from jet.scrapers.hrequests_utils import sync_scrape_url
from jet.transformers.formatters import format_json

PROMPT_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "generate a URL template for searching an anime on the website by replacing the specific anime title in the query parameter with '{{anime_title}}'. "
    "The template should be reusable for any anime title. "
    "Output only the URL template without any additional text then terminate immediately.\n"
    "Query: {query_str}\n"
    "Answer: "
)


class InvalidChoiceFormatError(Exception):
    """Raised when a choice does not match the expected format."""
    pass


def parse_choices(choices: List[str]) -> tuple[Dict[str, str], List[str]]:
    """
    Parses choices into a dictionary mapping keys to choice texts and a list of choice texts.
    Supports flexible key formats (e.g., 'A)', '1)', 'A.', 'A:') using regex.
    """
    key_to_choice = {}
    choice_texts = []
    # Regex to match keys (alphanumeric) followed by ')', '.', or ':'
    pattern = re.compile(r'^\s*([a-zA-Z0-9]+)[\)\.\:]\s*(.+?)\s*$')

    for choice in choices:
        if not choice.strip():
            raise InvalidChoiceFormatError(
                f"Choice '{choice}' is empty or invalid")

        match = pattern.match(choice)
        if not match:
            # For URLs, treat the entire choice as text if no key is present
            key_to_choice[str(len(choice_texts))] = choice.strip()
            choice_texts.append(choice.strip())
            continue

        key, text = match.groups()
        if not key or not text.strip():
            raise InvalidChoiceFormatError(
                f"Choice '{choice}' has empty key or text")

        if key in key_to_choice:
            raise InvalidChoiceFormatError(
                f"Duplicate key '{key}' found in choices")

        key_to_choice[key] = text.strip()
        choice_texts.append(text.strip())

    return key_to_choice, choice_texts


def calculate_token_probabilities(logprobs: List[float]) -> List[float]:
    """
    Convert log probabilities to regular probabilities.
    Assumes logprobs are natural logarithms.
    """
    return [math.exp(logprob) for logprob in logprobs]


def derive_choice_probability(generation_output: Dict, choice_keys: List[str]) -> Dict[str, float]:
    """
    Derive probabilities for each choice based on the first occurrence in the output.
    Args:
        generation_output: The JSON output containing logprobs and tokens.
        choice_keys: List of valid choice keys (e.g., ['1', '2', '3']).
    Returns:
        Dictionary mapping choice keys to their probabilities.
    """
    token_logprobs = generation_output['choices'][0]['logprobs']['token_logprobs']
    tokens = generation_output['choices'][0]['logprobs']['tokens']
    content = generation_output['choices'][0]['text'].strip().split('\n')

    # Initialize probabilities for each choice
    choice_probs = {key: 0.0 for key in choice_keys}
    found_choices = set()

    # Convert logprobs to probabilities
    probs = calculate_token_probabilities(token_logprobs)

    # Reconstruct text from tokens to map to content
    token_index = 0
    for token in generation_output['choices'][0]['text'].split('\n'):
        token = token.strip()
        if token:  # Skip empty tokens
            for key in choice_keys:
                if key not in found_choices and token in choice_keys:
                    choice_probs[key] = probs[token_index]
                    found_choices.add(key)
            token_index += 1
            # Skip newline
            while token_index < len(tokens) and tokens[token_index] == 198:
                token_index += 1

    # Normalize probabilities to sum to 1
    total_prob = sum(choice_probs.values())
    if total_prob > 0:
        choice_probs = {key: prob / total_prob for key,
                        prob in choice_probs.items()}

    return choice_probs


def generate_url_template(urls: List[str]) -> str:
    """
    Generate a URL template by extracting the common base URL and query parameter structure.
    """
    if not urls:
        return "https://aniwatchtv.to/search?keyword={anime_title}"

    # Assume all URLs follow the same structure, take the first one
    base_url = urls[0]
    # Extract the base URL up to 'keyword='
    match = re.search(r'^(https://aniwatchtv\.to/search\?keyword=)', base_url)
    if match:
        return f"{match.group(1)}{{anime_title}}"
    return "https://aniwatchtv.to/search?keyword={anime_title}"


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    search_results_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/mlx/generated/run_mlx_continuous_contexts_stream/browser_aniwatch_search_links_results.json"
    search_results: list[str] = load_file(search_results_file)["results"]
    anime_title = "Mushoku Tensei: Jobless Reincarnation"
    query = f"Generate a URL template for searching any anime on the website by replacing the specific anime title in the query parameter with '{{anime_title}}'. Use the provided search links as reference."

    formatted_docs = [
        search_result["url"]
        for search_result in search_results
    ]
    search_doc_results = search_docs(
        query=query,
        documents=formatted_docs,
        top_k=3
    )
    save_file(search_doc_results, f"{output_dir}/search_doc_results.json")

    urls = [doc["text"] for doc in search_doc_results]

    # Generate template directly from URLs
    template = generate_url_template(urls)
    save_file({"template": template}, f"{output_dir}/url_template.json")

    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    seed = 42
    mlx = MLX(model, seed=seed)

    prompt = PROMPT_TEMPLATE.format(
        query_str=query, context_str="\n".join(urls))
    generation_stream = mlx.stream_generate(
        model=model,
        verbose=True,
        prompt=prompt,
        logit_bias="https:",
        repetition_penalty=1.2,
        stop=["\n"]
    )
    generation_output = None
    url_template = ""
    for gen_response in generation_stream:
        content = gen_response["choices"][0].get("text", "")
        url_template += content

        if "\n" in content:
            break

        if gen_response["choices"][0]["finish_reason"]:
            generation_output = gen_response
            logger.newline()

    save_file(generation_output, f"{output_dir}/generation_output.json")
    save_file(url_template, f"{output_dir}/url_template.txt")

    # Check if anime is available in aniwatch
    titles_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/mlx/generated/run_mlx_continuous_contexts_stream/anime_titles.json"
    anime_titles: list[str] = load_file(titles_file)
    sub_dir = "title_search"
    output_title_dir = f"{output_dir}/{sub_dir}"
    for title in anime_titles:
        url = url_template.format(anime_title=title)
        logger.info(f"Checking {url}...")
        html_str = sync_scrape_url(url)
        if html_str:
            docs = get_md_header_docs(html_str)
            search_doc_results = search_docs(
                query=title,
                documents=[doc["header"].lstrip('#').strip() for doc in docs]
            )

            output_path = f"{output_title_dir}/{title}/search_doc_results.json"
            save_file({
                "query": title,
                "results": search_doc_results
            }, output_path)
