import os
import pandas as pd
import json
import more_itertools
from tqdm import tqdm
from multiprocessing import Pool
from transformers import pipeline

pool_count = 7
batch_limit = 100
data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
progress_file = "generated/classify_dataset/jobs_classifier_progress_info.json"
output_file = "generated/classify_dataset/jobs_classified.json"

# Classes for classification
classes = ["Web Developer", "Mobile Developer", "Full Stack Developer"]

# Classifier for zero-shot classification
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


def classify(sample):
    text = f"Job Details: {sample['details']}"

    # Classify the text
    classification = classifier(text, classes)

    # Add a category to the sample with the top classifier value
    sample['category'] = classification['labels'][0]
    return sample


def save_resume_info(resume_info):
    with open(progress_file, 'w') as f:
        json.dump(resume_info, f)


def load_resume_info():
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    else:
        return {"batch_index": 0}


def classify_data(dataframe, batch_limit=batch_limit):
    # Load resume info
    resume_info = load_resume_info()

    # All samples
    samples = dataframe.to_dict('records')

    # Initialize a multiprocessing Pool
    with Pool(pool_count) as p:
        # Split samples into batches
        for batch_index, batch in enumerate(more_itertools.chunked(samples, batch_limit)):
            # If this batch has been processed, skip
            if batch_index < resume_info['batch_index']:
                continue
            print(f'Processing batch {batch_index}...')

            # Classify samples in parallel and store in a list
            classified_samples = list(
                tqdm(p.imap_unordered(classify, batch), total=len(batch)))

            # Save the classified samples to a JSON file
            with open(output_file, 'a') as f:
                for sample in classified_samples:
                    json.dump(sample, f)
                    f.write("\n")

            print(f"Saved classified samples after batch")

            # Update resume info
            resume_info['batch_index'] = batch_index + 1
            save_resume_info(resume_info)


def convert_jsonl_to_json(jsonl_file_path, json_file_path):
    # Open the JSONL file for reading
    with open(jsonl_file_path, 'r') as file:
        # Read all lines from the JSONL file
        lines = file.readlines()

    # Create an empty list to store the JSON objects
    json_objects = [json.loads(line) for line in lines]

    # Open the JSON file for writing
    with open(json_file_path, 'w') as file:
        json.dump(json_objects, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # Load the JSON file
    df = pd.read_json(data_file)

    classify_data(df)
    convert_jsonl_to_json(output_file, output_file)
    print("Done!")
