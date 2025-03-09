from typing import Optional
import hrequests
from jet.data.utils import generate_unique_hash
from jet.llm.utils.embeddings import get_ollama_embedding_function
from jet.logger import logger
from jet.scrapers.browser.playwright import PageContent, scrape_sync, setup_sync_browser_page
from jet.scrapers.utils import extract_sentences
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.utils.object import extract_values_by_paths
from jet.wordnet.pos_tagger import POSTagger
from shared.data_types.job import JobData
from tqdm import tqdm
from jet.file.utils import save_file, load_file, load_data, save_data

from tqdm import tqdm
# from instruction_generator.helpers.dataset import load_data, save_data
# from instruction_generator.utils.time_utils import time_it
# from instruction_generator.translation.translator import Translator
from enum import Enum
from jet.logger import logger
# from instruction_generator.utils.language import detect_lang
# from instruction_generator.wordnet.pos_tagger import POSTagger


def process_batch(data: list[tuple(str, str)], tagger: POSTagger, tags: list[str], output_file: Optional[str] = None, batch_size=20, ):
    results = []

    pbar = tqdm(data)

    batch_counter = 0

    for idx, (sentence1, sentence2) in enumerate(pbar):
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']

        translation_text = None
        translation_text2 = None

        # expected = item.get('label')

        # has_label = expected in tags
        prev_result = None
        first_pos = None
        passed = False

        try:
            # Check if sentence ends with a question mark
            if sentence.strip()[-1] == "?":
                result = "Question"
            else:
                pos_results = tagger.process_and_tag(sentence)

                if pos_results[0]['pos'] in ['ADV', 'CCONJ', 'SCONJ']:
                    result = "Continuation"
                else:
                    result = "Statement"

            passed = result == expected

        except IndexError:
            result = "Error"
            passed = False
        finally:
            obj = {
                "id": item['id'],
                "sentence": sentence,
                "label": result,
            }
            # if has_label:
            #     obj['expected'] = expected
            #     obj['passed'] = passed
            #     obj['prev_result'] = prev_result

            results.append(obj)

            batch_counter += 1
            if batch_counter >= batch_size:
                if output_file:
                    save_data(output_file, results)
                batch_counter = 0

        if has_label:
            passed_count += 1 if passed else 0

            logger.info(
                f"\n\n---------- Item {idx + 1} ---------- Passed Percent: {passed_count / (idx + 1) * 100:.2f}%; {passed_count}/{idx + 1}")
            print("{:<35} {:<25} {:<15}".format(
                sentence[:30], "Actual: " + result, ""))
            print("{:<35} {:<25} {:<15}".format(
                translated_sentence[:30] if translated_sentence else "", "Expected: " + expected, "Previous: " + prev_result if prev_result else ""))
            logger.success(
                f"Passed: {passed}") if passed else logger.error("Failed")

            print("\n")


def main():
    embed_model = "mxbai-embed-large"
    json_attributes = ["title", "details"]

    # Load job data
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    output_file = "generated/sentence_intents.json"

    data: list[JobData] = load_file(data_file) or []

    texts_to_embed = []
    for item in tqdm(data):
        json_parts_dict = extract_values_by_paths(
            item, json_attributes, is_flattened=True) if json_attributes else None
        text_parts = []
        for key, value in json_parts_dict.items():
            value_str = str(value)
            if isinstance(value, list):
                value_str = ", ".join(value)
            if value_str.strip():
                text_parts.append(
                    f"{key.title().replace('_', ' ')}: {value_str}")
        text_content = "\n".join(text_parts) if text_parts else ""

        texts_to_embed.append(text_content)

    embed_func = get_ollama_embedding_function(embed_model)
    embed_results = embed_func(texts_to_embed)


if __name__ == "__main__":
    main()
