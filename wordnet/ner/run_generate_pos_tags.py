import random
import json
import os
import shutil
from jet.file.utils import load_file, save_file
from jet.wordnet.pos_tagger import POSTagger
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


class BatchSaver:
    def __init__(self, output_file, batch_size=200):
        self.output_file = output_file
        self.batch_size = batch_size
        self.data_batch = []
        self.existing_texts = self.load_existing()

    def load_existing(self, unique_key='text'):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if unique_key:
                    return {item[unique_key] for item in existing_data if unique_key in item}
                else:
                    return existing_data
        return set()

    def add_item(self, item, unique_key='text'):
        text = item.get(unique_key, None)
        if text and text not in self.existing_texts:
            self.data_batch.append(item)
            self.existing_texts.add(text)
            if len(self.data_batch) >= self.batch_size:
                self.save_and_reset()

    def save_and_reset(self):
        save_file(self.data_batch, self.output_file)
        self.data_batch = []

    def save_remaining(self):
        if self.data_batch:
            self.save_and_reset()


def tag_json_files(data, tagger, output_file):
    logging.info("Starting POS tagging process...")
    random.Random(42).shuffle(data)
    batch_saver = BatchSaver(output_file)
    existing_texts = batch_saver.load_existing()

    with tqdm(total=len(data), desc="Tagging texts", unit="text") as pbar:
        for text_value in data:
            if text_value and text_value not in existing_texts:
                pos_results = tagger.process_and_tag(text_value)
                tagged_results_dict = {
                    'text': text_value,
                    'lang': 'en',
                    'pos': pos_results,
                    'pos_text': tagger.format_tags(pos_results)
                }
                batch_saver.add_item(tagged_results_dict)
                existing_texts.add(text_value)
            pbar.update(1)

    batch_saver.save_remaining()
    logging.info(f"Completed POS tagging. Results saved to {output_file}")


def tag_json_files_pos_word_counts(data, tagger, output_file):
    logging.info("Starting POS word count processing...")
    random.Random(42).shuffle(data)
    pos_word_counts = {'en': {}}
    batch_size = 200
    batch_count = 0

    with tqdm(total=len(data), desc="Counting POS words", unit="text") as pbar:
        for idx, text_value in enumerate(data):
            batch_count += 1
            if text_value:
                pos_results = tagger.process_and_tag(text_value, lang='en')
                for pair in pos_results:
                    pos = pair['pos']
                    word = pair['word']
                    lower_word = word.lower()
                    pos_word_counts_lang = pos_word_counts['en']
                    if pos not in pos_word_counts_lang:
                        pos_word_counts_lang[pos] = {}
                    if lower_word not in pos_word_counts_lang[pos]:
                        pos_word_counts_lang[pos][lower_word] = 0
                    pos_word_counts_lang[pos][lower_word] += 1
            pbar.update(1)

            if batch_count >= batch_size or idx == len(data) - 1:
                for pos in pos_word_counts['en']:
                    pos_word_counts['en'][pos] = dict(
                        sorted(pos_word_counts['en'][pos].items(), key=lambda item: item[0]))
                save_file(pos_word_counts, output_file)
                batch_count = 0

    logging.info(
        f"Completed POS word count processing. Results saved to {output_file}")
    return pos_word_counts


def main():
    logging.info("Starting dataset processing...")
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/headers.json"
    logging.info(f"Loading dataset from {data_file}")
    data: list[dict] = load_file(data_file)
    logging.info(f"Loaded {len(data)} items from dataset")
    data: list[str] = [d["content"] for d in data]
    tagger = POSTagger()

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    output_file = f"{output_dir}/tagged_data.json"
    tag_json_files(data, tagger, output_file)


if __name__ == "__main__":
    main()
