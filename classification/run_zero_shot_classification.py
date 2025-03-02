import os
from typing import Any, TypedDict
import pandas as pd
import json
from jet.logger import logger
import more_itertools
from shared.data_types.job import JobData
from tqdm import tqdm
from multiprocessing import Pool
from transformers import pipeline
from llama_index.core.schema import BaseNode, Document, TextNode
from jet.file import load_file
from jet.utils.object import extract_values_by_paths


data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
progress_file = "generated/classify_dataset/jobs_classifier_progress_info.json"
output_file = "generated/classify_dataset/jobs_classified.json"
pool_count = 7
batch_limit = 100


class ClassificationResponse(TypedDict):
    labels: list[str]
    scores: list[float]
    sequence: str


classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


def classify(text: str, classes: list[str]) -> ClassificationResponse:
    """
    Classify the text using zero-shot classification.

    :param text: The text to be classified
    :param classes: The list of possible labels for classification
    :return: Classification results including labels, scores, and the input text
    """
    # Classify the text
    classification: ClassificationResponse = classifier(text, classes)

    # Cast the result to the ClassificationResponse type
    return ClassificationResponse(
        labels=classification["labels"],
        scores=classification["scores"],
        sequence=text
    )


if __name__ == '__main__':
    # Classes for classification
    classes_steps = [
        ["Web developer only", "Mobile developer only", "Web and mobile developer"],
        ["Full stack web developer", "Full stack mobile developer"]
    ]

    metadata_attributes = [
        "id",
        "title",
        "details",
        "link",
        "company",
        "posted_date",
        "salary",
        "job_type",
        "hours_per_week",
        "domain",
        "tags",
        "keywords",
        "entities.role",
        "entities.application",
        "entities.coding_libraries",
        "entities.qualifications",
    ]

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)

    filtered_data = [d for d in data if d['id'] in ["1316960-onlinejobs.ph"]]
    data = filtered_data

    docs: list[Document] = []
    for item in data:
        title = item['title']
        details = item['details']

        textdata_str = f"Details:\n{details}"
        if title not in details:
            textdata_str = f"Job Title: {title}\n{textdata_str}"

        metadata = extract_values_by_paths(
            item, metadata_attributes, is_flattened=True)

        docs.append(Document(
            text=textdata_str,
            metadata=metadata,
        ))

    for doc_idx, doc in enumerate(tqdm(docs)):
        title = doc.metadata['title']
        keywords = doc.metadata['keywords']
        tags = doc.metadata['tags']
        role = doc.metadata.get('role')
        application = doc.metadata.get('application')
        coding_libraries = doc.metadata.get('coding_libraries')
        qualifications = doc.metadata.get('qualifications')
        details = doc.metadata['details']

        texts = [
            f"Job Title: {title}",
            f"Keywords: {", ".join(keywords)}"
            f"Tags: {", ".join(tags)}"
        ]
        if role:
            texts.append(f"Role: {", ".join(role)}")
        if application:
            texts.append(f"Application: {", ".join(application)}")
        if coding_libraries:
            texts.append(f"Technology stack: {", ".join(coding_libraries)}")
        if qualifications:
            texts.append(f"Qualifications: {", ".join(qualifications)}")

        texts.append(f"Job Details:\n{details}")

        text = "\n".join(texts)

        classes_accumulated_results = []
        for classes in classes_steps:
            # updated_classes = classes_accumulated_results + classes
            updated_classes = classes
            classification = classify(text, updated_classes)
            top_label = classification["labels"][0]
            classes_accumulated_results.append(top_label)

        final_results = classes_accumulated_results
        logger.debug(f"Doc {idx + 1} results:")
        logger.success(final_results)
