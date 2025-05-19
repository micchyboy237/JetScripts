from dataset import load_data, generate_unique_id
from sentence import adaptive_split
from tqdm import tqdm
import json


def main():
    batch = 500

    data_en_file = 'data/alpaca_gpt4_data_en.json'
    data_tl_file = 'data/alpaca_gpt4_data_ph.json'

    data_en = load_data(data_en_file)
    data_tl = load_data(data_tl_file)

    split_en_tl_data = []
    max_segment_tokens = 60

    pbar = tqdm(total=len(data_tl), desc="Splitting data")

    # Loop through each data point of data_tl
    for idx, data in enumerate(data_tl):
        # Get the corresponding data point from data_en and data_tl
        data_en_item = data_en[idx]
        data_tl_item = data

        # split data_en_item and data_tl_item instruction and output
        instruction_en = data_en_item['instruction']
        instruction_tl = data_tl_item['instruction']
        input_en = data_en_item['input']
        input_tl = data_tl_item['input']
        output_en = data_en_item['output']
        output_tl = data_tl_item['output']

        # Split the instructions and outputs into segments
        instruction_en_segments = adaptive_split(instruction_en)
        instruction_tl_segments = adaptive_split(instruction_tl)
        input_en_segments = adaptive_split(input_en)
        input_tl_segments = adaptive_split(input_tl)
        output_en_segments = adaptive_split(output_en)
        output_tl_segments = adaptive_split(output_tl)

        en_segments = instruction_en_segments + input_en_segments + output_en_segments
        tl_segments = instruction_tl_segments + input_tl_segments + output_tl_segments

        # Strip all segments
        en_segments = [segment.strip() for segment in en_segments]
        tl_segments = [segment.strip() for segment in tl_segments]

        # # Check if the number of segments are the same
        if len(en_segments) != len(tl_segments):
            pbar.update(1)

            continue

        split_en_tl_item = {
            'id': generate_unique_id(),
            'en': en_segments,
            'tl': tl_segments
        }

        split_en_tl_data.append(split_en_tl_item)

        # Save the split data to a file on every batch
        if ((idx + 1) % batch) == 0 or idx + 1 == len(data_tl):
            with open('instruction_generator/datasets/split_alpaca_gpt4_data_en_ph.json', 'w', encoding='utf-8') as file:
                json.dump(split_en_tl_data, file, indent=4, ensure_ascii=False)

        pbar.update(1)


if __name__ == "__main__":
    main()
