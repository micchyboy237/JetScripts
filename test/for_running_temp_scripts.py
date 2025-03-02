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


def extract_entities(body: dict):
    response = http_client.post(
        f"{NER_API_BASE_URL}/extract-entities", json=body)
    response.raise_for_status()

    raw_content = response.text.strip()
    logger.debug(f"Raw Response: {raw_content}")

    try:
        # Split multiple JSON objects and parse them separately
        json_objects = [json.loads(line)
                        for line in raw_content.split("\n") if line.strip()]
        return json_objects  # Returns a list of parsed JSON objects
    except json.decoder.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e}")
        logger.error(f"Response Content: {raw_content}")
        raise


def main():
    # Load job data
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file) or []

    data_with_rn = [
        d for d in data
        if "React Native" in d["keywords"]
        or "React Native" in d["entities"]["coding_libraries"]
    ]

    logger.success(len(data_with_rn))


if __name__ == "__main__":
    main()
