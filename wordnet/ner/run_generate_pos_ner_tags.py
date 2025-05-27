import json
import os
from jet.data.utils import generate_key

from jet.wordnet.sentence import adaptive_split
from jet.wordnet.utils import get_content_from_url, is_valid_path
from tqdm import tqdm
from jet.file.utils import load_data, save_data

import re
import random

keys_to_process = ['result_en', 'result_en2', 'result_tl', 'result_tl2']
labels_dict = {
    'Location': 'LOC',
    'Person': 'PER',
    'Organization': 'ORG'
}


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
        save_data(self.output_file, self.data_batch)
        self.data_batch = []

    def save_remaining(self):
        if self.data_batch:
            self.save_and_reset()


def calculate_scores_percent_difference(top_score, second_score):
    return 100 * (top_score - second_score) / top_score


def generate_url(lang, word):
    return f"https://{lang}.wikipedia.org/wiki/{word}"


def process_ner_label(data):
    from jet.wordnet.pos_ner_tagger import POSTaggerProperNouns

    batch_size = 2
    model = POSTaggerProperNouns()
    batch_saver = BatchSaver(
        'server/static/models/dost-asti-gpt2/base_model/datasets/base/wiki_ner.json', batch_size)
    existing_items = batch_saver.load_existing(unique_key=None)
    existing_item_subjects = [item['subject'] for item in existing_items]
    data = [item for item in data if item['title']
            not in existing_item_subjects]

    en_data = [item for item in data if item['language'] == 'English']

    random.Random().shuffle(en_data)

    ner_data = []
    all_subjects = [item['subject'] for item in existing_items]
    labels_dict_reversed = {v: k for k, v in labels_dict.items()}

    pbar = tqdm(en_data, desc=f"Processing Data...")

    for item in pbar:
        lang = 'en' if item['language'] == 'English' else 'tl'
        subject = item['title']
        en_url = generate_url('en', subject)
        tl_url = generate_url('tl', subject)

        en_id = generate_key(en_url)
        tl_id = generate_key(tl_url)

        if subject in all_subjects:
            continue

        all_subjects.append(subject)

        pbar.set_description_str(f"Url: {en_url}")
        en_content_result = get_content_from_url(en_url)
        if not en_content_result:
            continue

        pbar.set_description_str(f"Url: {tl_url}")
        tl_content_result = get_content_from_url(tl_url)
        if not tl_content_result:
            continue

        en_label = None
        tl_label = None

        en_title = en_content_result['title']
        if not is_valid_path(en_title):
            continue
        en_content = en_content_result['content']
        first_sentence = adaptive_split(en_content)[0]
        pos_ner_results = model.predict(first_sentence, lang=lang)

        for item in pos_ner_results:
            matched_pos_words = re.findall(rf'\b{en_title}\b', item['span'])
            if matched_pos_words:
                en_label = labels_dict_reversed[item['label']]
                break

        tl_title = tl_content_result['title']
        if not is_valid_path(tl_title):
            continue
        tl_content = tl_content_result['content']
        first_sentence = adaptive_split(tl_content)[0]
        pos_ner_results = model.predict(first_sentence, lang=lang)

        for item in pos_ner_results:
            matched_pos_words = re.findall(rf'\b{tl_title}\b', item['span'])
            if matched_pos_words:
                tl_label = labels_dict_reversed[item['label']]
                break

        if (not en_label and not tl_label) or (en_label and tl_label and en_label == tl_label):

            if en_content_result and tl_content_result:
                en_url = en_content_result['url']
                tl_url = tl_content_result['url']

                en_obj = {
                    'id': en_id,
                    'lang': 'en',
                    'subject': subject,
                    'title': en_title,
                    'label': en_label,
                    'url': en_url,
                    'content': en_content
                }

                tl_obj = {
                    'id': tl_id,
                    'lang': 'tl',
                    'subject': subject,
                    'title': tl_title,
                    'label': tl_label,
                    'url': tl_url,
                    'content': tl_content
                }

                ner_data.append(en_obj)
                ner_data.append(tl_obj)

                batch_saver.add_item(en_obj, unique_key='id')
                batch_saver.add_item(tl_obj, unique_key='id')

                print(
                    f"Current batch size: {len(batch_saver.data_batch)}")

    batch_saver.save_remaining()

    return ner_data


def main():
    # base_data = load_data(
    #     'server/static/datasets/content/wiki_initial_descriptions.json')
    base_data = [
        {"language": "English", "title": "Manila"},
        {"language": "English", "title": "Ferdinand Marcos"},
        {"language": "English", "title": "University of the Philippines"}
    ]

    process_ner_label(base_data)


if __name__ == "__main__":
    main()
