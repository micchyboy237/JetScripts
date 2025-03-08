import json
from jet.logger import logger
from jet.memory.httpx import HttpxClient
from jet.scrapers.utils import clean_text
from jet.vectors.ner import merge_dot_prefixed_words
from shared.data_types.job import JobData
from tqdm import tqdm
from jet.file.utils import save_file, load_file

NER_API_BASE_URL = "http://0.0.0.0:8002/api/v1/ner"

http_client = HttpxClient()


def extract_entity(body: dict):
    response = http_client.post(
        f"{NER_API_BASE_URL}/extract-entity", json=body)
    response.raise_for_status()
    return response.json()


def main():
    # Load job data
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file) or []

    labels = ["role", "application", "technology stack", "qualifications"]
    chunk_sizes = [250]

    base_keywords = ['React.js', 'React Native']

    my_skills_keywords = {
        "React Native": "React Native",
        "React": "React.js",
        "Node": "Node.js",
        "Python": "Python",
        "PostgreSQL": "PostgreSQL",
        "MongoDB": "MongoDB",
        "Firebase": "Firebase",
        "AWS": "AWS",
    }

    results = []

    for chunk_size in chunk_sizes:
        for item in tqdm(data):
            text = f"{item['title']}\n\n{item['details']}"

            # Extract technology stack from entities
            extracted_tech = extract_entity({
                "labels": labels,
                "chunk_size": chunk_size,
                "text": text,
            }).get('technology_stack', [])

            # Normalize keywords while preserving mapped values
            normalized_tech = {}
            current_text = text
            sorted_reverse_keywords = sorted(
                my_skills_keywords.keys(), reverse=True)
            for keyword in sorted_reverse_keywords:
                if keyword in current_text:
                    normalized_tech[keyword.lower(
                    )] = my_skills_keywords[keyword]
                current_text = current_text.replace(keyword, "")
            # Add extracted technologies while preserving case
            for tech in extracted_tech:
                # Keep first occurrence case, prefer mapped values if available
                normalized_tech.setdefault(
                    tech.lower(), my_skills_keywords.get(tech, tech))
            # Get unique values
            technology_stack = list(
                set(map(merge_dot_prefixed_words, normalized_tech.values())))
            matched_skill_keywords = list({
                t for t in technology_stack
                if t in list(my_skills_keywords.values())
            })

            # Clean & update technology stack
            item['keywords'] = matched_skill_keywords
            item['entities']['technology_stack'] = technology_stack

            active_keywords = [
                base_keyword for base_keyword in base_keywords
                if base_keyword in item['keywords']
            ]

            if active_keywords:
                results.append(item)

        save_file(data, data_file)


if __name__ == "__main__":
    main()
