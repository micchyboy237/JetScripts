"""
job_chunker.py

Chunking pipeline for jobs.json-style records.

Strategy:
- Each job is treated as one retrieval unit by default (jobs are already
  short, self-contained, and splitting across jobs would destroy meaning).
- A "context header" built from the structured fields (title, company,
  salary, job_type, tags, domain, hours_per_week) is prepended to every
  chunk so no chunk loses track of which job / role / pay it belongs to.
- Only the `details` field is split, and only when it's long enough to
  exceed max_tokens. Splitting happens on paragraph boundaries first,
  falling back to sentence boundaries, never mid-sentence.
- Structured fields (domain, job_type, salary, tags) are also kept as
  separate metadata so they can be used for hybrid filtering at query
  time instead of relying purely on embeddings to "understand" them.

No external dependencies required (token estimate is character-based,
~4 chars/token, which is a standard rough approximation for English).
Swap `estimate_tokens` for a real tokenizer (e.g. tiktoken) if you want
exact counts.
"""

import json
import logging
import re
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("job_chunker")

# ---- Tunables -----------------------------------------------------------

MAX_TOKENS = 400  # target max tokens per chunk (excluding header)
OVERLAP_TOKENS = 60  # ~15% overlap between consecutive sub-chunks
CHARS_PER_TOKEN = 4  # rough estimate, no external tokenizer needed


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (chars / 4). Good enough for chunk sizing."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def build_context_header(job: dict[str, Any]) -> str:
    """
    Build the metadata header prepended to every chunk for this job.
    This is what keeps a lone fragment of `details` meaningful on its own.
    """
    parts = [f"Job: {job.get('title', 'Unknown title')}"]

    if job.get("company"):
        parts.append(f"Company: {job['company']}")

    if job.get("salary"):
        parts.append(f"Salary: {job['salary']}")

    if job.get("job_type"):
        parts.append(f"Type: {job['job_type']}")

    if job.get("hours_per_week"):
        parts.append(f"Hours/week: {job['hours_per_week']}")

    if job.get("tags"):
        parts.append(f"Tags: {', '.join(job['tags'])}")

    parts.append(f"Source: {job.get('domain', 'unknown')}")

    header = " | ".join(parts)
    logger.debug(f"[{job.get('id')}] Built header: {header}")
    return header


def split_long_details(details: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split long `details` text into sub-chunks on paragraph boundaries.
    Falls back to sentence boundaries within a paragraph if a single
    paragraph itself exceeds max_tokens. Never cuts mid-sentence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", details) if p.strip()]

    # Further split any oversized "paragraph" into sentences
    units: list[str] = []
    for para in paragraphs:
        if estimate_tokens(para) <= max_tokens:
            units.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            units.extend(s.strip() for s in sentences if s.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)

        if current_tokens + unit_tokens > max_tokens and current:
            chunks.append(" ".join(current))

            # Build overlap: carry trailing units forward until overlap budget spent
            overlap_units: list[str] = []
            overlap_tok = 0
            for u in reversed(current):
                t = estimate_tokens(u)
                if overlap_tok + t > overlap_tokens:
                    break
                overlap_units.insert(0, u)
                overlap_tok += t

            current = overlap_units
            current_tokens = overlap_tok

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_job(
    job: dict[str, Any],
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """
    Produce one or more retrieval-ready chunks for a single job record.
    Returns a list of dicts: {chunk_id, text, metadata}
    """
    job_id = job.get("id", "unknown")
    details = (job.get("details") or "").strip()
    header = build_context_header(job)

    base_metadata = {
        "job_id": job_id,
        "link": job.get("link"),
        "title": job.get("title"),
        "company": job.get("company"),
        "posted_date": job.get("posted_date"),
        "domain": job.get("domain"),
        "salary": job.get("salary"),
        "job_type": job.get("job_type"),
        "hours_per_week": job.get("hours_per_week"),
        "tags": job.get("tags", []),
    }

    details_tokens = estimate_tokens(details)

    if details_tokens <= max_tokens:
        logger.info(f"[{job_id}] Single chunk ({details_tokens} est. tokens)")
        return [
            {
                "chunk_id": f"{job_id}_0",
                "text": f"{header}\n\n{details}" if details else header,
                "metadata": {**base_metadata, "chunk_index": 0, "chunk_count": 1},
            }
        ]

    logger.info(f"[{job_id}] Long details ({details_tokens} est. tokens) -> splitting")
    sub_chunks = split_long_details(
        details, max_tokens - estimate_tokens(header), overlap_tokens
    )

    result = []
    for i, sub in enumerate(sub_chunks):
        result.append(
            {
                "chunk_id": f"{job_id}_{i}",
                "text": f"{header}\n\n{sub}",
                "metadata": {
                    **base_metadata,
                    "chunk_index": i,
                    "chunk_count": len(sub_chunks),
                },
            }
        )

    logger.info(f"[{job_id}] Produced {len(result)} sub-chunks")
    return result


def process_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run chunk_job over the full jobs.json list, logging summary stats."""
    all_chunks: list[dict[str, Any]] = []
    single_chunk_count = 0
    split_count = 0

    for job in jobs:
        chunks = chunk_job(job)
        all_chunks.extend(chunks)
        if len(chunks) == 1:
            single_chunk_count += 1
        else:
            split_count += 1

    logger.info(
        f"Done. {len(jobs)} jobs -> {len(all_chunks)} chunks "
        f"({single_chunk_count} single-chunk jobs, {split_count} jobs split further)"
    )
    return all_chunks


if __name__ == "__main__":
    jobs_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/saved/jobs.json"

    # Example usage
    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs_data = json.load(f)

    chunks = process_jobs(jobs_data)

    with open("jobs_chunked.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info("Wrote jobs_chunked.json")
