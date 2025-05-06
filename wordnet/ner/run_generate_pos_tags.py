import random
import json
import os
import shutil
from jet.file.utils import load_file, save_file
from jet.wordnet.pos_tagger import POSTagger
from tqdm import tqdm


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
    random.Random(42).shuffle(data)
    batch_saver = BatchSaver(output_file)
    existing_texts = batch_saver.load_existing()

    for text_value in tqdm(data, desc="Processing..."):
        if text_value and text_value not in existing_texts:
            pos_results = tagger.process_and_tag(text_value)
            tagged_results_dict = {
                'lang': 'en',
                'text': text_value,
                'pos_text': tagger.format_tags(pos_results),
                'pos': pos_results,
            }
            batch_saver.add_item(tagged_results_dict)
            existing_texts.add(text_value)

    batch_saver.save_remaining()


def tag_json_files_pos_word_counts(data, tagger, output_file):
    random.Random(42).shuffle(data)
    pos_word_counts = {'en': {}}
    batch_size = 200
    batch_count = 0

    for idx, text_value in enumerate(tqdm(data, desc="Processing counts...")):
        batch_count += 1
        if text_value:
            pos_results = tagger.process_and_tag(text_value)
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

        if batch_count >= batch_size or idx == len(data) - 1:
            for pos in pos_word_counts['en']:
                pos_word_counts['en'][pos] = dict(
                    sorted(pos_word_counts['en'][pos].items(), key=lambda item: item[0]))
            save_file(pos_word_counts, output_file)
            batch_count = 0

    return pos_word_counts


def main():
    print("Loading dataset...")
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/headers.json"
    data: list[dict] = load_file(data_file)
    data: list[str] = [d["content"] for d in data]
    tagger = POSTagger()

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/tagged_data.json"

    tag_json_files(data, tagger, output_file)


if __name__ == "__main__":
    main()
