import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import tiktoken

# ========================= CONFIG =========================
CHUNK_SIZE = 512  # Tokens - good default for most embedding models
CHUNK_OVERLAP = 80  # Tokens - helps preserve context
LOG_LEVEL = logging.INFO
# =======================================================

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_token_count(text: str) -> int:
    """Accurate token counting using tiktoken (cl100k_base is common)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def load_jobs(file_path: str = "jobs.json") -> List[Dict]:
    """Load jobs from JSON file (list or NDJSON)."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File {file_path} not found. Create it with your job data.")
        raise FileNotFoundError(f"File {file_path} not found.")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            jobs = json.loads(content)
        else:
            # NDJSON fallback
            jobs = [json.loads(line) for line in content.splitlines() if line.strip()]
    logger.info(f"Loaded {len(jobs)} jobs from {file_path}")
    return jobs


def enrich_chunk_text(job: Dict, details_chunk: str) -> str:
    """Create rich text for embedding - metadata + content."""
    return (
        f"Job Title: {job.get('title', 'N/A')}\n"
        f"Company: {job.get('company', 'N/A')}\n"
        f"Domain: {job.get('domain', 'N/A')}\n"
        f"Job Type: {job.get('job_type', 'N/A')} | Salary: {job.get('salary', 'N/A')}\n"
        f"Tags: {', '.join(job.get('tags', []))}\n"
        f"Posted: {job.get('posted_date', 'N/A')}\n\n"
        f"Details:\n{details_chunk}"
    )


def chunk_job(
    job: Dict, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """Chunk a single job with metadata enrichment."""
    job_id = job.get("id")
    details = (job.get("details") or "").strip()

    if not details:
        logger.warning(f"Job {job_id} has no details.")
        chunk_text = enrich_chunk_text(job, f"Title: {job.get('title')}")
        return [{**job, "chunk_text": chunk_text, "chunk_id": f"{job_id}_full"}]

    token_count = get_token_count(details)
    logger.debug(f"Job {job_id} details tokens: {token_count}")

    # Short job -> single chunk
    if token_count <= int(chunk_size * 1.5):
        chunk_text = enrich_chunk_text(job, details)
        return [{**job, "chunk_text": chunk_text, "chunk_id": f"{job_id}_full"}]

    # Recursive splitting for longer details
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=get_token_count,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "• ",
            " ",
            "",
        ],  # Job-friendly (paragraphs, bullets)
    )

    detail_chunks = splitter.split_text(details)
    enriched_chunks = []

    for i, chunk in enumerate(detail_chunks):
        chunk_text = enrich_chunk_text(job, chunk)
        enriched = {
            **job,  # Full original metadata
            "chunk_text": chunk_text,  # For embedding
            "chunk_id": f"{job_id}_{i}",
            "chunk_index": i,
            "total_chunks": len(detail_chunks),
            "original_details_tokens": token_count,
        }
        enriched_chunks.append(enriched)

    logger.info(f"Job {job_id} split into {len(detail_chunks)} chunks")
    return enriched_chunks


def chunk_all_jobs(jobs: List[Dict]) -> List[Dict]:
    """Process all jobs."""
    all_chunks = []
    for job in jobs:
        try:
            chunks = chunk_job(job)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error chunking job {job.get('id')}: {e}")
    logger.info(f"Total chunks created: {len(all_chunks)} from {len(jobs)} jobs")
    return all_chunks


def save_chunks(chunks: List[Dict], output_path: str = "job_chunks.json"):
    """Save chunks for later use / inspection."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


# ========================= DEMO / USAGE =========================
if __name__ == "__main__":
    # Sample data for quick testing (remove when using real jobs.json)
    sample_jobs = [
        {
            "id": "1591976",
            "link": "https://www.onlinejobs.ph/jobseekers/job/1591976",
            "title": "Part-Time Programming Tutor (JavaScript, React, Django) - 5-10 hrs/week",
            "company": "",
            "posted_date": "2026-06-20T19:31:26",
            "details": "I am currently enrolled in an online Computer Science program and am looking for a part-time virtual programming tutor to supplement my coursework.\n\nThis role is focused on structured learning and ...",
            "domain": "onlinejobs.ph",
            "salary": "Negotiable",
            "job_type": "Any",
            "hours_per_week": 10,
            "tags": ["Javascript", "React JS", "Python"],
        },
        # Add more from your file...
    ]

    jobs = load_jobs() if Path("jobs.json").exists() else sample_jobs
    chunks = chunk_all_jobs(jobs)
    save_chunks(chunks)

    # Preview first chunk
    if chunks:
        print("\n=== SAMPLE CHUNK PREVIEW ===")
        sample = chunks[0]
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"Title: {sample.get('title')}")
        print(f"Tokens in chunk_text: {get_token_count(sample['chunk_text'])}")
        print("Chunk text preview:")
        print(sample["chunk_text"][:500] + "...")
