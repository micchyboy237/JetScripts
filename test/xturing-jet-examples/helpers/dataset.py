import os
import json
import numpy as np
import pickle
from jet.logger import time_it
from words import get_words, get_non_words, count_words
from collections import Counter, defaultdict
from typing import List, Optional, Tuple, Dict, Any
import uuid
import hashlib
import fnmatch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def generate_unique_id():
    # Generate a unique id
    return uuid.uuid4().hex


def generate_hash(item, max_length=24):
    """Generate a consistent, truncated hash for a given item."""
    item_str = json.dumps(
        item, sort_keys=True)  # Convert item to a string with sorted keys
    hash_key = hashlib.sha256(item_str.encode()).hexdigest()
    return hash_key[:max_length]


def load_dataset_from_huggingface_to_json(dataset_name, output_file):
    # Customizing tqdm to display a progress bar for the download and load process
    # tqdm.pandas()

    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Convert the dataset to a pandas DataFrame
    # Assuming you want to work with the 'train' split, change this if necessary
    df = pd.DataFrame(dataset['train'])

    # Save the DataFrame to a JSON file
    df.to_json(output_file, orient='records', lines=False)


def load_data_from_directories(source_directories, includes=None, excludes=None):
    data = []

    for directory in source_directories:
        # Check if directory is a json file
        if os.path.isfile(directory) and directory.endswith(".json"):
            source_file = directory
            with open(source_file, 'r') as file:
                data.extend(json.load(file))
            continue
        for filename in os.listdir(directory):
            # Apply include and exclude filters
            if (not includes or any(fnmatch.fnmatch(filename, pattern) for pattern in includes)) and \
               (not excludes or not any(fnmatch.fnmatch(filename, pattern) for pattern in excludes)):
                source_file = os.path.join(directory, filename)
                data.extend(load_data(source_file))

    return data


def load_data(file_path: str, is_binary=False):
    has_no_extension = not os.path.splitext(file_path)[1]
    if has_no_extension or file_path.endswith(".bin"):
        is_binary = True

    if is_binary:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    elif file_path.endswith(".csv"):
        data = pd.read_csv(file_path).to_dict(orient='records')
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
    elif not os.path.isdir(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = load_data_from_directories([file_path])

    return data


def save_data(output_file, data, write=False, key='id', is_binary=False):
    if not data:
        print(f"No data to save for {output_file}")
        return
    # Check if the output file has no extension
    has_no_extension = not os.path.splitext(output_file)[1]
    if has_no_extension or output_file.endswith(".bin"):
        is_binary = True

    if write or not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # data = [dict(t) for t in {tuple(d.items()) for d in data}]

        print(f"Writing {len(data)} items to {output_file}")

        if is_binary:
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        # Update existing data array with matching data array based on item['key]
        updated_data_dict = {
            item[key]: item for item in existing_data if key in item}
        for idx, item in enumerate(data):
            if item.get(key, None) in updated_data_dict:
                existing_data_index = next(
                    (i for i, x in enumerate(existing_data) if x[key] == item[key]), None)
                existing_data[existing_data_index] = {
                    **existing_data[existing_data_index],
                    **item
                }
            else:
                existing_data.append(item)

        # Deduplicate by key
        # existing_data = [dict(t) for t in {tuple(d.items()) for d in existing_data}]
        print(f"Writing {len(existing_data)} items to {output_file}")

        if is_binary:
            with open(output_file, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)


def is_mostly_words(text):
    # Get words and non-words from the text
    words = get_words(text)
    non_words = get_non_words(text)

    # Check if non-words exceed the threshold count of 50
    if len(non_words) > 50:
        return False

    if len(words) <= len(non_words):
        return False

    # Calculate the total number of tokens and the difference threshold
    total_tokens = len(words) + len(non_words)
    diff_length = len(words) - len(non_words)
    difference_threshold = 15  # 15%
    difference_percentage = diff_length / total_tokens
    difference_percentage = difference_percentage * 100

    if difference_percentage < difference_threshold:
        return False

    return True


def filter_dataset_with_mostly_words(data, keys):
    filtered_data = []

    pbar = tqdm(data, desc="Filtering")
    for item in pbar:
        text = " ".join([item[key] for key in keys])
        if is_mostly_words(text):
            filtered_data.append(item)

        pbar.set_postfix({"Filtered": len(filtered_data)})

    return filtered_data


class ProcessedData():
    source: str
    target: Optional[str]
    category_values: List[str]
    score: Optional[float]


@time_it
def load_translation_data(
    file_path_or_dirs,
    includes=None,
    excludes=None,
    max_q=2,
    source_key='translation.en',
    target_key='translation.tl',
    score_key='translation.score'
) -> List[ProcessedData]:

    return load_samples(
        file_path_or_dirs,
        includes,
        excludes,
        max_q,
        source_key,
        target_key,
        score_key,
    )


@time_it
def load_samples(
    file_path_or_dirs=None,
    includes=None,
    excludes=None,
    max_q=2,
    source_key='translation.en',
    target_key='translation.tl',
    score_key='translation.score',
    data=[]
) -> List[ProcessedData]:
    if not data:
        if isinstance(file_path_or_dirs, str):
            data = load_data(file_path_or_dirs)
        else:
            data = load_data_from_directories(
                file_path_or_dirs, includes, excludes)

    def calculate_ttr(sentence):
        words = get_words(sentence)
        unique_words = set(words)
        return len(unique_words)

    def calculate_ttr_class(ttr, ttr_quantiles):
        for i, q in enumerate(ttr_quantiles):
            if ttr <= q:
                return f'ttr_q{i+1}'
        return f'ttr_q{len(ttr_quantiles)+1}'

    def categorize_sentence_length(sentence, length_quantiles):
        word_count = count_words(sentence)
        for i, q in enumerate(length_quantiles):
            if word_count <= q:
                return f'q{i+1}'
        return f'q{len(length_quantiles)+1}'

    def determine_quantiles(values, num_quantiles):
        """ Determine dynamic quantile values based on the data distribution """
        quantile_values = np.linspace(0, 1, num_quantiles + 2)[1:-1]
        return np.quantile(values, quantile_values)

    # Compute quantiles for dynamic categorization
    sentence_counts = [len(item[source_key].split()) for item in data]
    ttrs = [calculate_ttr(item[source_key]) for item in data]

    # Determine number of quantiles based on data diversity and max_q
    num_length_quantiles = min(max_q, min(5, len(set(sentence_counts)) // 20))
    num_ttr_quantiles = min(max_q, min(5, len(set(ttrs)) // 20))

    length_quantiles = determine_quantiles(
        sentence_counts, num_length_quantiles)
    ttr_quantiles = determine_quantiles(ttrs, num_ttr_quantiles)

    processed_data = []
    ttr_class_distribution = Counter()
    sentence_length_distribution = Counter()

    for item in data:
        source_sentence = item[source_key]
        target_sentence = item.get(target_key, None)
        score = item.get(score_key, None)

        ttr = calculate_ttr(source_sentence)
        ttr_class = calculate_ttr_class(ttr, ttr_quantiles)
        sentence_length = categorize_sentence_length(
            source_sentence, length_quantiles)

        # Update distributions
        ttr_class_distribution[ttr_class] += 1
        sentence_length_distribution[sentence_length] += 1

        # Create a new processed data item
        processed_item = ProcessedData()
        processed_item.source = source_sentence
        processed_item.target = target_sentence
        processed_item.category_values = [ttr_class, sentence_length]
        processed_item.score = score

        processed_data.append(processed_item)

    # Print out the distributions
    print("TTR Class Distribution:", dict(ttr_class_distribution))
    print("Sentence Length Distribution:", dict(sentence_length_distribution))

    return processed_data


def load_unique_samples(data: List[Dict], seed: int = 42) -> List[Dict]:
    """Load unique samples ensuring one label per sample, given a dataset. Handles unbalanced data."""
    try:
        stratify_labels = [d['label'] for d in data]
    except KeyError as exc:
        raise ValueError(
            "Please ensure that all samples have a 'label' key for stratification.") from exc

    label_counts = Counter(stratify_labels)

    # Check if it's possible to stratify (i.e., more than one sample per label)
    if any(count == 1 for count in label_counts.values()):
        # If any label has only one sample, return all unique samples
        # This also covers the case where there is only one label
        unique_samples = [data[i] for i, label in enumerate(
            stratify_labels) if label_counts[label] == 1]
        return unique_samples

    # If all labels have more than one sample, perform stratified split
    unique_labels = len(set(stratify_labels))
    print(f"Unique labels: {unique_labels}")
    train_data, _ = train_test_split(
        data, train_size=unique_labels, random_state=seed, stratify=stratify_labels)
    return train_data


def extract_word_texts(
        words,
        texts,
        text_per_word_limit=5,
        min_word_length=3,
        max_word_length=15,
        exclude_words=[],
        from_start=False):

    words_texts_dict = {}
    added_texts = []

    pbar = tqdm(texts, postfix="Extracting word texts")
    for text in pbar:
        text = text.strip()
        if not text:
            continue
        text_words = get_words(text)
        failed_min_words = len(text_words) < min_word_length
        failed_max_words = len(text_words) > max_word_length
        first_letter_is_not_upper = text[0].islower()
        not_ends_with_punctuation = text[-1] not in ['.', '!', '?']
        # has_non_alpha = any(not word.isalpha() for word in text_words)
        if failed_min_words \
                or failed_max_words \
                or first_letter_is_not_upper \
                or not_ends_with_punctuation:
            continue

        text_words = get_words(text)
        text_2_words = get_words(text, 2)
        all_words = text_words + text_2_words
        all_words = sorted(all_words, key=len, reverse=True)
        added_words = []
        for word in all_words:
            # if from_start and text startswith word
            first_word = text_words[0]
            not_text_startswith_word = from_start and first_word != word.capitalize()
            text_exists = text in added_texts
            if word in added_words \
                    or word not in words \
                    or word in exclude_words \
                    or not_text_startswith_word \
                    or text_exists:
                continue

            if not word in words_texts_dict:
                words_texts_dict[word] = []
                pbar.set_description_str(
                    f"Added words: {len(words_texts_dict)}")

            if text not in words_texts_dict[word]:
                words_texts_dict[word].append(text)
                added_words.append(word)
                added_texts.append(text)

    for word in tqdm(words, desc="Limiting word texts"):
        word_texts = words_texts_dict.get(word, [])

        if word_texts:
            # apply unique
            word_texts = list(set(word_texts))
            # sort by length of text in ascending order
            # word_texts = sorted(word_texts, key=len)
            # apply limit
            word_texts = word_texts[:text_per_word_limit]
            words_texts_dict[word] = word_texts

    return words_texts_dict


def distribute_evenly(data: List[Dict[str, Any]], key: str = 'label') -> List[Dict[str, Any]]:
    grouped_items = defaultdict(list)
    for item in data:
        grouped_items[item[key]].append(item)

    # Sort items by label count in descending order
    grouped_items = dict(sorted(grouped_items.items(),
                         key=lambda x: len(x[1]), reverse=True))

    # Interleave items to spread out identical labels
    distributed = []
    while grouped_items:
        keys_to_delete = []
        for key in grouped_items:
            distributed.append(grouped_items[key].pop(0))
            if not grouped_items[key]:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del grouped_items[key]

    return distributed
