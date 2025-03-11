import os

from jet.file.utils import load_file
from jet.wordnet.histogram import generate_histogram


if __name__ == "__main__":
    # data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/wordnet/data/histogram_data_sample.json"
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data = load_file(data_file) or []

    output_dir = f'/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/wordnet/data/generated/jobs_histogram'

    include_keys = ['title', 'details']

    generate_histogram(data_file, output_dir, include_keys=include_keys)
