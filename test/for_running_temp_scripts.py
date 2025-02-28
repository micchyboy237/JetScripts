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

    # Filter data by ids
    exclude_ids = ['1324241']
    data = [d for d in data if d['id'] in exclude_ids]

    labels = ["role", "application", "technology stack", "qualifications"]
    chunk_sizes = [250]

    my_skills_keywords = [
        "React",
        "React Native",
        "Node",
        "Python",
        "PostgreSQL",
        "MongoDB",
        "Firebase",
        "AWS",
    ]

    for chunk_size in chunk_sizes:
        for item in tqdm(data):
            text = f"{item['title']}\n\n{item['details']}"

            # Preserve case from my_skills_keywords
            normalized_tech = {skill.lower(
            ): skill for skill in my_skills_keywords if skill.lower() in text.lower()}

            # Extract technology stack from entities
            extracted_tech = extract_entity({
                "labels": labels,
                "chunk_size": chunk_size,
                "text": text,
            }).get('technology_stack', [])

            # Add extracted technologies while preserving case
            for tech in extracted_tech:
                # Keep first occurrence case
                normalized_tech.setdefault(tech.lower(), tech)

            # Clean & update technology stack
            item['entities']['technology_stack'] = list(
                set(map(merge_dot_prefixed_words, normalized_tech.values())))

        save_file(data, data_file)

    # # Calling extract_entity
    # labels = ["technology stack"]
    # chunk_size = 250
    # sample_text = "Job Title:\n\n???? Full Stack Web Developer – Data & Analytics Platform (TikTok Shop Integration)\n\n\n\nJob Description:\n\nWe are looking for a skilled Full Stack Web Developer to help us replicate and build a website similar to \nUpgrade to see actual info\n. The ideal candidate should have experience in building data-driven websites, API integration, and user-friendly dashboards.\n\n\n\nThis role requires someone who can work independently and build a functional, scalable platform that collects and presents analytics, particularly related to TikTok Shop performance data.\n\n\n\nResponsibilities:\n\n? Develop a fully functional web platform similar to \nUpgrade to see actual info\n\n? Frontend Development: Design & develop a clean, modern UI/UX using React, Vue, or Angular\n\n? Backend Development: Build a scalable system using Node.js, Python (Django/Flask), or PHP (Laravel)\n\n? Database Management: Set up and manage a database (MySQL, PostgreSQL, or MongoDB) for storing user and analytics data\n\n? API Integration: Connect and integrate with TikTok Shop APIs & third-party analytics APIs\n\n? User Dashboard & Reports: Create a user-friendly dashboard displaying data insights\n\n? Performance Optimization: Ensure website is fast, secure, and scalable\n\n\n\nRequirements:\n\n???? 3+ years of experience in full-stack web development\n\n???? Proficiency in React, Vue, or Angular (Frontend)\n\n???? Strong backend skills in Node.js, Python, or PHP (Laravel)\n\n???? Experience with API integration (TikTok, eCommerce, or Analytics APIs preferred)\n\n???? Database management with MySQL, PostgreSQL, or MongoDB\n\n???? Familiarity with web scraping (if needed for analytics data collection)\n\n???? Ability to create visually appealing dashboards & reports\n\n???? Experience working with analytics tools & data visualization\n\n???? Strong problem-solving skills & ability to work independently\n\n\n\nBonus Skills (Not Required but Preferred):\n\n? Experience with AI/ML for data analytics\n\n? Knowledge of Cloud Hosting (AWS, DigitalOcean, or Firebase)\n\n? Background in eCommerce analytics or TikTok Shop data\n\n\n\nMESSAGE ME IF YOU HAVE ANY QUESTIONS OR COMMENTS!"
    # cleaned_text = clean_text(sample_text)
    # single_request_body = {
    #     "labels": labels,
    #     "chunk_size": chunk_size,
    #     "text": cleaned_text,
    # }
    # entity_result = extract_entity(single_request_body)
    # logger.debug(f"Entity:")
    # logger.success(format_json(entity_result))

    # # Calling extract_entities
    # labels = ["technology stack"]
    # chunk_size = 512
    # chunk_overlap = 128
    # set_global_tokenizer(lemmatize_text)
    # splitter = SentenceSplitter(
    #     chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=lemmatize_text)
    # cleaned_text_chunks = splitter.split_text(cleaned_text)
    # multi_request_body = {
    #     "labels": labels,
    #     "chunk_size": chunk_size,
    #     "data": [{"text": text} for text in cleaned_text_chunks],
    # }
    # entities_result = extract_entities(multi_request_body)
    # logger.debug(f"Batch Entities ({len(entities_result)}):")
    # logger.success(format_json(entities_result))


if __name__ == "__main__":
    main()
