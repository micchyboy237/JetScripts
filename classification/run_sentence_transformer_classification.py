from jet.file.utils import save_file
from jet.utils.object import extract_values_by_paths
from jet.file import load_file
from llama_index.core.schema import Document
from tqdm import tqdm
from shared.data_types.job import JobData
from jet.logger import logger
from typing import Any, TypedDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, TypedDict


data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
model = 'all-MiniLM-L12-v2'


class ClassificationResponse(TypedDict):
    results: List[tuple[str, float]]  # List of (label, score) pairs
    sequence: str


# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def classify(text: str, classes: List[str], score_threshold: float = 0.5) -> ClassificationResponse:
    """
    Classify the text by calculating cosine similarity to each label using sentence embeddings.
    Filters out results with a score below the given threshold and sorts results by score in descending order.

    :param text: The text to be classified
    :param classes: The list of possible labels for classification
    :param score_threshold: The minimum score for a label to be included in the results
    :return: Classification results including (label, score) pairs and the input text
    """
    # Compute embeddings for the text and labels
    text_embedding = model.encode([text])
    label_embeddings = model.encode(classes)

    # Compute cosine similarities between the text and each label
    similarities = cosine_similarity(text_embedding, label_embeddings)[0]

    # Filter out results with a score below the threshold
    filtered_results = [
        (label, score) for label, score in zip(classes, similarities) if score >= score_threshold
    ]

    # Sort results by score in descending order
    sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

    # Return the response with sorted results and sequence
    return ClassificationResponse(
        results=sorted_results,
        sequence=text
    )


if __name__ == '__main__':
    # Classes for classification
    classes = ["Web developer", "Mobile developer", "Backend developer"]
    score_threshold = 0.25

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
    output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/classified_jobs.json"
    data: list[JobData] = load_file(data_file)

    # filtered_data = [d for d in data if d['id'] in ["1316960-onlinejobs.ph"]]
    # data = filtered_data

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

    results = []
    results_dict = {
        "results": results
    }
    for doc_idx, doc in enumerate(tqdm(docs)):
        id = doc.metadata['id']
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

        classification = classify(text, classes, score_threshold)
        classification_results = classification["results"]
        final_results = [
            {"label": label, "score": score}
            for label, score in classification_results
        ]
        results.append({
            "id": id,
            "text": text,
            "classification": final_results
        })

        logger.debug(f"Doc {doc_idx + 1} results ({len(results)}):")

        save_file(results_dict, output_file)
