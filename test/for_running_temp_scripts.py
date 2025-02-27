import json

from jet.logger import logger
from jet.memory.httpx import HttpxClient
from typing import List, Set
from jet.scrapers.utils import clean_text
from shared.data_types.job import JobEntity
from jet.transformers import format_json
from llama_index.core.utils import set_global_tokenizer
from shared.data_types.job import JobData
from tqdm import tqdm
from jet.file.utils import save_file, load_file

NER_API_BASE_URL = "http://0.0.0.0:8002/api/v1/ner"
NER_MODEL = "urchade/gliner_small-v2.1"

http_client = HttpxClient()


def extract_entity(body: dict) -> list[str]:
    response = http_client.post(
        f"{NER_API_BASE_URL}/extract-entity", json=body)
    response.raise_for_status()
    return response.json()


def main():
    sample_text = "**Job Type:** Full-Time / Project-Based (Remote, PH Applicants Only)\n**Salary:** Competitive (Negotiable Based on Experience)\n---\n### **Job Description:**\nWe seek a skilled developer to build an AI-powered recipe generator, enhancing features beyond DishGen. You will develop the AI backend, intuitive UI/UX, and advanced meal customization options for a premium SaaS platform.\n---\n### **Responsibilities:**\n- Develop an AI-driven recipe generator with personalized meal plans.\n- Build a responsive, mobile-friendly interface.\n- Optimize AI algorithms for speed and accuracy.\n- Implement authentication, payment processing, and database management.\n- Ensure performance, security, and scalability.\n---\n### **Requirements:**\n- Full-stack development (Python, JavaScript, React/Vue, HTML, CSS).\n- Experience with AI/ML (OpenAI API, TensorFlow, NLP).\n- Cloud services (AWS, Firebase) & database management.\n- Strong problem-solving skills & ability to work independently.\n---\n### **Why Join Us?**\n- Work on an exciting AI-driven SaaS project.\n- Competitive salary + performance bonuses.\n- Flexible remote work setup.\n---\n### **How to Apply:**\nSend your resume, portfolio, and a short cover letter. If you've built similar projects, share your work!"
    sample_text = clean_text(sample_text)
    labels = ["technology stack"]
    chunk_sizes = [256, 512, 768, 1024]

    entities_all_results: Set[str] = set()
    entities_size_results = {}

    progress_bar = tqdm(chunk_sizes, unit="chunk", dynamic_ncols=True)

    for chunk_size in progress_bar:
        progress_bar.set_description(f"Processing chunk (size: {chunk_size})")

        single_request_body = {
            "model": NER_MODEL,
            "labels": labels,
            "chunk_size": chunk_size,
            "text": sample_text,
        }
        entity_result = extract_entity(single_request_body)
        technology_stack = entity_result["technology_stack"]

        # Add unique entities to the set
        extracted_entities = set(technology_stack)
        entities_all_results.update(extracted_entities)
        entities_size_results[chunk_size] = list(extracted_entities)

        logger.debug(
            f"Chunk ({chunk_size}) | Entities ({len(extracted_entities)}):")

    logger.success(format_json(list(entities_all_results)))
    logger.newline()
    logger.debug(f"All Entities for chunks {chunk_sizes}:", len(
        entities_all_results), colors=["DEBUG", "SUCCESS"])
    logger.success(format_json(entities_size_results))


if __name__ == "__main__":
    main()
