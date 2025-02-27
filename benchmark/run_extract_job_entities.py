import json
from jet.logger import logger
from jet.memory.httpx import HttpxClient
from jet.scrapers.utils import clean_text
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from pydantic import BaseModel
from typing import List
from shared.data_types.job import JobEntity
from jet.transformers import format_json

NER_API_BASE_URL = "http://0.0.0.0:8002/api/v1/ner"

http_client = HttpxClient()


class TextRequest(BaseModel):
    text: str


class ProcessRequest(BaseModel):
    model: str = "urchade/gliner_small-v2.1"
    labels: List[str]
    style: str = "ent"
    data: List[TextRequest]
    chunk_size: int = 250


class SingleTextRequest(BaseModel):
    text: str
    model: str = "urchade/gliner_small-v2.1"
    labels: List[str]
    style: str = "ent"
    chunk_size: int = 250


class ProcessedTextResponse(BaseModel):
    text: str
    entities: JobEntity


class ProcessResponse(BaseModel):
    data: List[ProcessedTextResponse]


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
    labels = ["technology stack"]
    chunk_size = 1024
    chunk_overlap = 128
    sample_text = "Job Title:\n\nFull Stack Web Developer – Data & Analytics Platform (TikTok Shop Integration)\n\n\n\nJob Description:\n\nWe are looking for a skilled Full Stack Web Developer to help us replicate and build a website similar to \nUpgrade to see actual info\n. The ideal candidate should have experience in building data-driven websites, API integration, and user-friendly dashboards.\n\n\n\nThis role requires someone who can work independently and build a functional, scalable platform that collects and presents analytics, particularly related to TikTok Shop performance data.\n\n\n\nResponsibilities:\n\nDevelop a fully functional web platform similar to \nUpgrade to see actual info\n\nFrontend Development: Design & develop a clean, modern UI/UX using React, Vue, or Angular\n\nBackend Development: Build a scalable system using Node.js, Python (Django/Flask), or PHP (Laravel)\n\nDatabase Management: Set up and manage a database (MySQL, PostgreSQL, or MongoDB) for storing user and analytics data\n\nAPI Integration: Connect and integrate with TikTok Shop APIs & third-party analytics APIs\n\nUser Dashboard & Reports: Create a user-friendly dashboard displaying data insights\n\nPerformance Optimization: Ensure website is fast, secure, and scalable\n\n\n\nRequirements:\n\n3+ years of experience in full-stack web development\n\nProficiency in React, Vue, or Angular (Frontend)\n\nStrong backend skills in Node.js, Python, or PHP (Laravel)\n\nExperience with API integration (TikTok, eCommerce, or Analytics APIs preferred)\n\nDatabase management with MySQL, PostgreSQL, or MongoDB\n\nFamiliarity with web scraping (if needed for analytics data collection)\n\nAbility to create visually appealing dashboards & reports\n\nExperience working with analytics tools & data visualization\n\nStrong problem-solving skills & ability to work independently\n\n\n\nBonus Skills (Not Required but Preferred):\n\nExperience with AI/ML for data analytics\n\nKnowledge of Cloud Hosting (AWS, DigitalOcean, or Firebase)\n\nBackground in eCommerce analytics or TikTok Shop data\n\n\n\nMESSAGE ME IF YOU HAVE ANY QUESTIONS OR COMMENTS!"

    tokenizer =

    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    cleaned_text = clean_text(sample_text)
    cleaned_text_chunks = splitter.split_text(cleaned_text)

    # Calling extract_entity
    single_request_body = {
        "labels": labels,
        "chunk_size": chunk_size,
        "text": cleaned_text,
    }
    entity_result = extract_entity(single_request_body)
    logger.debug(f"Entity:")
    logger.success(format_json(entity_result))

    # Calling extract_entities
    multi_request_body = {
        "labels": labels,
        "chunk_size": chunk_size,
        "data": [{"text": text} for text in cleaned_text_chunks],
    }
    entities_result = extract_entities(multi_request_body)
    logger.debug(f"Batch Entities ({len(entities_result)}):")
    logger.success(format_json(entities_result))


if __name__ == "__main__":
    main()
