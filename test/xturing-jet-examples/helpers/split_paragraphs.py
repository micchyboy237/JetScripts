from dataset import load_data, generate_unique_id
from words import count_words
from tqdm import tqdm
import json


def main():
    batch = 500

    data_file = 'server/static/datasets/content/wiki_initial_descriptions.json'

    data = load_data(data_file)

    # Filter data by language == 'English'
    # get item "content" attribute
    data_en = [item['content']
               for item in data if item['language'] == 'English']
    data_tl = [item['content']
               for item in data if item['language'] == 'Tagalog']

    split_en_tl_data = []
    min_words = 10

    pbar = tqdm(total=len(data_tl), desc="Splitting paragraphs")

    for idx, data in enumerate(data_en):
        # Split data by newline
        paragraphs = data.split('\n')

        # Loop through paragraphs
        for paragraph in paragraphs:
            # Count words in paragraph
            words = count_words(paragraph)

            # If paragraph has more than min_words words
            if words > min_words:
                # Generate unique id
                unique_id = generate_unique_id()

                # Append to split data
                split_en_tl_data.append({
                    'id': unique_id,
                    'en': paragraph,
                    'tl': ''
                })

        # Save the split data to a file on every batch
        if ((idx + 1) % batch) == 0 or idx + 1 == len(data_tl):
            with open('instruction_generator/datasets/wiki_paragraphs_en_ph.json', 'w', encoding='utf-8') as file:
                json.dump(split_en_tl_data, file, indent=4, ensure_ascii=False)

        pbar.update(1)


if __name__ == "__main__":
    main()
