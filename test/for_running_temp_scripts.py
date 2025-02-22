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
WEB_DEV_KEYWORDS = ["React.js", "React",
                    "Web app", "Web developer", "Web development"]
MOBILE_DEV_KEYWORDS = ["React Native", "Mobile app",
                       "Mobile developer", "Mobile development", "iOS", "Android"]
BACKEND_DEV_KEYWORDS = ["Node.js", "Node", "Firebase", "AWS", "Backend"]


class ClassificationResponse(TypedDict):
    results: List[tuple[str, float]]  # List of (label, score) pairs
    sequence: str


# Initialize the sentence transformer model
model = SentenceTransformer(model)


def validate_dev_type(words: list[str], base_keywords: list[str]) -> list:
    # Return a list of matching words found in the job description
    return [word for word in words if any(base_keyword.lower() == word.lower() for base_keyword in base_keywords)]


def classify(job: JobData) -> dict:
    title = job['title']
    keywords = job['keywords']
    entities = job['entities']
    role = entities.get('role', [])
    technology_stack = entities.get('technology_stack', [])
    application = entities.get('application', [])
    qualifications = entities.get('qualifications', [])

    classification_dict = {
        "web": [],
        "mobile": [],
        "backend": [],
        "labels": []  # Add a labels attribute
    }

    # words_to_validate = title.split(' ') + keywords + technology_stack + \
    #     role + application + qualifications
    words_to_validate = keywords + technology_stack

    # Get matching words for each category
    web_matches = validate_dev_type(words_to_validate, WEB_DEV_KEYWORDS)
    mobile_matches = validate_dev_type(words_to_validate, MOBILE_DEV_KEYWORDS)
    backend_matches = validate_dev_type(
        words_to_validate, BACKEND_DEV_KEYWORDS)

    # Update classification dictionary with unique matching words
    if web_matches:
        classification_dict["web"] = list(
            set(web_matches))  # Remove duplicates
        classification_dict["labels"].append("Web developer")
    if mobile_matches:
        classification_dict["mobile"] = list(
            set(mobile_matches))  # Remove duplicates
        classification_dict["labels"].append("Mobile developer")
    if backend_matches:
        classification_dict["backend"] = list(
            set(backend_matches))  # Remove duplicates
        classification_dict["labels"].append("Backend developer")

    return classification_dict


if __name__ == '__main__':
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/classified_jobs.json"
    jobs: list[JobData] = load_file(data_file)

    results = []
    results_dict = {
        "results": results
    }

    for job in jobs:
        classification_labels = classify(job)

        results.append({
            "id": job['id'],
            "link": job['link'],
            "title": job['title'],
            "classification": classification_labels,
            "technology_stack": job['entities'].get('technology_stack', [])
        })

    save_file(results_dict, output_file)
