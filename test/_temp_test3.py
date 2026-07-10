"""
Job Postings RAG Demo
=====================

A complete, reusable demo for RAG on structured job posting data.
Uses document-level chunking (1 job = 1 chunk) with rich metadata filtering.

Features:
- Loads jobs.json with optional fields
- Document-level chunking (no splitting)
- FAISS vector store for local demo
- Hybrid search: metadata filtering + semantic search
- Reusable RAG pipeline class

Requirements:
    pip install langchain sentence-transformers faiss-cpu numpy

Usage:
    python job_postings_rag_demo.py
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class JobPosting:
    """Structured representation of a job posting."""

    id: str
    link: str
    title: str
    company: str
    posted_date: str
    details: str
    domain: str
    salary: Optional[str] = None
    job_type: Optional[str] = None
    hours_per_week: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobPosting":
        """Create JobPosting from dictionary (handles missing fields)."""
        return cls(
            id=str(data["id"]),
            link=str(data.get("link", "")),
            title=str(data.get("title", "")),
            company=str(data.get("company", "")),
            posted_date=str(data.get("posted_date", "")),
            details=str(data.get("details", "")),
            domain=str(data.get("domain", "")),
            salary=data.get("salary"),
            job_type=data.get("job_type"),
            hours_per_week=data.get("hours_per_week"),
            tags=data.get("tags", []),
        )

    def to_embedding_text(self) -> str:
        """Create text for embedding (title + company + details)."""
        parts = [self.title]
        if self.company:
            parts.append(f"\n\nCompany: {self.company}")
        if self.details:
            parts.append(f"\n\n{self.details}")
        return "".join(parts)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dictionary."""
        return {
            "id": self.id,
            "link": self.link,
            "title": self.title,
            "company": self.company,
            "posted_date": self.posted_date,
            "domain": self.domain,
            "salary": self.salary,
            "job_type": self.job_type,
            "hours_per_week": self.hours_per_week,
            "tags": self.tags,
        }


class JobPostingRAG:
    """
    Reusable RAG pipeline for job postings.

    Uses document-level chunking (1 job = 1 chunk) with metadata filtering.

    Args:
        embedding_model: SentenceTransformer model or None (uses default)
        vector_store_path: Path to save/load FAISS index
        jobs_file: Path to jobs.json file
    """

    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        vector_store_path: str = "job_postings_index",
        jobs_file: str = "jobs.json",
    ):
        self.embedding_model_name = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.vector_store_path = vector_store_path
        self.jobs_file = jobs_file
        self.job_postings: List[JobPosting] = []
        self.embedding_model = None
        self.index = None
        self.job_id_to_index: Dict[str, int] = {}
        self.metadata: List[Dict[str, Any]] = []

        self._validate_requirements()
        self._load_embedding_model()

    def _validate_requirements(self):
        """Check if required packages are installed."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )
        if not HAS_FAISS:
            raise ImportError("faiss-cpu required. Install with: pip install faiss-cpu")

    def _load_embedding_model(self):
        """Load the embedding model."""
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def load_jobs(self, file_path: Optional[str] = None) -> List[JobPosting]:
        """Load job postings from JSON file."""
        path = file_path or self.jobs_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Jobs file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            jobs_data = json.load(f)

        if isinstance(jobs_data, dict):
            jobs_data = [jobs_data]

        self.job_postings = [JobPosting.from_dict(job) for job in jobs_data]
        print(f"Loaded {len(self.job_postings)} job postings")
        return self.job_postings

    def create_embeddings(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Create embeddings for all job postings."""
        texts = [job.to_embedding_text() for job in self.job_postings]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        self.metadata = [job.to_metadata() for job in self.job_postings]
        self.job_id_to_index = {
            job.id: idx for idx, job in enumerate(self.job_postings)
        }

        return embeddings.astype("float32"), self.metadata

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def save_index(self, index: faiss.Index, embeddings: np.ndarray):
        """Save FAISS index and metadata to disk."""
        os.makedirs(self.vector_store_path, exist_ok=True)

        faiss.write_index(index, os.path.join(self.vector_store_path, "index.faiss"))

        with open(os.path.join(self.vector_store_path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)

        with open(os.path.join(self.vector_store_path, "id_mapping.json"), "w") as f:
            json.dump(self.job_id_to_index, f)

        print(f"Index and metadata saved to: {self.vector_store_path}")

    def load_index(self) -> Tuple[faiss.Index, np.ndarray, List[Dict[str, Any]]]:
        """Load FAISS index and metadata from disk."""
        index_path = os.path.join(self.vector_store_path, "index.faiss")
        metadata_path = os.path.join(self.vector_store_path, "metadata.json")
        mapping_path = os.path.join(self.vector_store_path, "id_mapping.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at: {index_path}")

        index = faiss.read_index(index_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        with open(mapping_path, "r") as f:
            job_id_to_index = json.load(f)

        self.metadata = metadata
        self.job_id_to_index = {k: int(v) for k, v in job_id_to_index.items()}

        return index, None, metadata

    def initialize(self, rebuild: bool = False) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            rebuild: If True, rebuild the index from scratch
        """
        index_path = os.path.join(self.vector_store_path, "index.faiss")
        if not rebuild and os.path.exists(index_path):
            print("Loading existing index...")
            self.index, _, self.metadata = self.load_index()
            return

        print("Loading jobs and creating embeddings...")
        self.load_jobs()
        embeddings, metadata = self.create_embeddings()

        print("Building FAISS index...")
        self.index = self.build_index(embeddings)
        self.save_index(self.index, embeddings)

        print("RAG pipeline initialized successfully!")

    def filter_by_metadata(
        self,
        domain: Optional[str] = None,
        posted_after: Optional[str] = None,
        posted_before: Optional[str] = None,
        tags: Optional[List[str]] = None,
        job_type: Optional[str] = None,
        salary_min: Optional[float] = None,
        hours_per_week: Optional[int] = None,
    ) -> List[int]:
        """
        Filter job postings by metadata.

        Returns list of indices that match the filters.
        """
        matching_indices = list(range(len(self.metadata)))

        if domain:
            matching_indices = [
                i for i in matching_indices if self.metadata[i].get("domain") == domain
            ]

        if posted_after:
            try:
                cutoff_date = datetime.fromisoformat(
                    posted_after.replace("Z", "+00:00")
                )
                matching_indices = [
                    i
                    for i in matching_indices
                    if datetime.fromisoformat(
                        self.metadata[i]["posted_date"].replace("Z", "+00:00")
                    )
                    >= cutoff_date
                ]
            except (ValueError, TypeError):
                pass

        if posted_before:
            try:
                cutoff_date = datetime.fromisoformat(
                    posted_before.replace("Z", "+00:00")
                )
                matching_indices = [
                    i
                    for i in matching_indices
                    if datetime.fromisoformat(
                        self.metadata[i]["posted_date"].replace("Z", "+00:00")
                    )
                    <= cutoff_date
                ]
            except (ValueError, TypeError):
                pass

        if tags:
            matching_indices = [
                i
                for i in matching_indices
                if any(
                    tag.lower() in [t.lower() for t in self.metadata[i].get("tags", [])]
                    for tag in tags
                )
            ]

        if job_type:
            matching_indices = [
                i
                for i in matching_indices
                if str(job_type).lower()
                in str(self.metadata[i].get("job_type", "")).lower()
            ]

        if salary_min is not None:
            matching_indices = [
                i
                for i in matching_indices
                if self._extract_salary_min(self.metadata[i].get("salary", ""))
                >= salary_min
            ]

        if hours_per_week is not None:
            matching_indices = [
                i
                for i in matching_indices
                if self.metadata[i].get("hours_per_week") == hours_per_week
            ]

        return matching_indices

    def _extract_salary_min(self, salary_str: str) -> float:
        """Extract minimum salary from salary string."""
        if not salary_str:
            return 0.0

        if "-" in salary_str or "–" in salary_str:
            parts = salary_str.replace("-", "–").split("–")
            if parts:
                first_part = parts[0].strip()
                import re

                match = re.search(r"[\d,]+(?:\.\d+)?", first_part)
                if match:
                    return float(match.group().replace(",", ""))

        import re

        match = re.search(r"[\d,]+(?:\.\d+)?", salary_str)
        if match:
            return float(match.group().replace(",", ""))

        return 0.0

    def semantic_search(
        self,
        query: str,
        k: int = 10,
        domain: Optional[str] = None,
        posted_after: Optional[str] = None,
        posted_before: Optional[str] = None,
        tags: Optional[List[str]] = None,
        job_type: Optional[str] = None,
        salary_min: Optional[float] = None,
        hours_per_week: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search: metadata filtering + semantic search.

        Args:
            query: Search query text
            k: Number of results to return
            domain: Filter by domain (onlinejobs.ph, ph.jobstreet.com, linkedin.com)
            posted_after: Filter by minimum posted date (ISO format)
            posted_before: Filter by maximum posted date (ISO format)
            tags: Filter by tags (list of strings)
            job_type: Filter by job type
            salary_min: Filter by minimum salary
            hours_per_week: Filter by hours per week

        Returns:
            List of matching job postings with scores
        """
        filtered_indices = self.filter_by_metadata(
            domain=domain,
            posted_after=posted_after,
            posted_before=posted_before,
            tags=tags,
            job_type=job_type,
            salary_min=salary_min,
            hours_per_week=hours_per_week,
        )

        if not filtered_indices:
            return []

        query_embedding = self.embedding_model.encode([query])

        if len(filtered_indices) == 1:
            idx = filtered_indices[0]
            score = float(np.linalg.norm(query_embedding - self.index.reconstruct(idx)))
            return [
                {
                    **self.metadata[idx],
                    "score": 1.0 / (1.0 + score),
                    "text": self.job_postings[idx].to_embedding_text(),
                }
            ]

        filtered_embeddings = np.array(
            [self.index.reconstruct(i) for i in filtered_indices]
        )

        distances = np.linalg.norm(filtered_embeddings - query_embedding, axis=1)

        top_k_indices_in_filtered = np.argsort(distances)[
            : min(k, len(filtered_indices))
        ]

        results = []
        for rank, filtered_idx in enumerate(top_k_indices_in_filtered):
            original_idx = filtered_indices[filtered_idx]
            distance = distances[filtered_idx]
            results.append(
                {
                    **self.metadata[original_idx],
                    "score": 1.0 / (1.0 + float(distance)),
                    "rank": rank + 1,
                    "text": self.job_postings[original_idx].to_embedding_text(),
                }
            )

        return results

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job posting by its ID."""
        if job_id not in self.job_id_to_index:
            return None

        idx = self.job_id_to_index[job_id]
        return {
            **self.metadata[idx],
            "text": self.job_postings[idx].to_embedding_text(),
        }


def create_sample_jobs_json():
    """Create a sample jobs.json file for testing."""
    sample_jobs = [
        {
            "id": "1591976",
            "link": "https://www.onlinejobs.ph/jobseekers/job/1591976",
            "title": "Part-Time Programming Tutor (JavaScript, React, Django) - 5-10 hrs/week",
            "company": "Private Student",
            "posted_date": "2026-06-20T19:31:26",
            "details": "I am currently enrolled in an online Computer Science program and am looking for a part-time virtual programming tutor to supplement my coursework. This role is focused on structured learning and hands-on coding practice. You should be proficient in JavaScript, React, and Django. The position requires 5-10 hours per week, flexible schedule. Payment is negotiable based on experience.",
            "domain": "onlinejobs.ph",
            "salary": "Negotiable",
            "job_type": "Part-time",
            "hours_per_week": 10,
            "tags": ["Javascript", "React JS", "Python", "Django", "Tutoring"],
        },
        {
            "id": "92840537",
            "link": "https://ph.jobstreet.com/job/92840537",
            "title": "AI Agents & Automation Engineer",
            "company": "Online Helpers",
            "posted_date": "2026-06-20T13:47:38",
            "details": "We're hiring a hands-on builder who can create real systems using AI tools. You'll design and deploy AI agents, automations, and LLM-powered workflows that replace manual processes and support business operations. Experience with Python, LangChain, and vector databases is required. Remote work with flexible hours.",
            "domain": "ph.jobstreet.com",
            "salary": "$600 - $850 per month (USD)",
            "job_type": "Full time",
            "hours_per_week": 40,
            "tags": ["AI", "Automation", "LLM", "LangChain", "Python"],
        },
        {
            "id": "4429633048",
            "link": "https://www.linkedin.com/jobs/view/4429633048",
            "title": "Senior Software Engineer (Full Stack)",
            "company": "Aphex",
            "posted_date": "2026-06-19T00:00:00",
            "details": "About Aphex: We're the construction planning platform that's replacing outdated spreadsheets with multiplayer tools that delivery teams love. Major contractors like BAM, Balfour Beatty, SKANSKA, Kier use our platform. We're looking for a Senior Full Stack Engineer to join our growing team. Experience with React, Node.js, and cloud platforms required.",
            "domain": "linkedin.com",
            "salary": "$5000 - $7000 per month (USD)",
            "job_type": "Full-time",
            "hours_per_week": None,
            "tags": ["React", "Node.js", "Full Stack", "Cloud", "Construction Tech"],
        },
        {
            "id": "77788899",
            "link": "https://www.onlinejobs.ph/jobseekers/job/77788899",
            "title": "Backend Developer (Python/Django)",
            "company": "Tech Startup PH",
            "posted_date": "2026-06-21T10:00:00",
            "details": "We are a fast-growing tech startup in Manila looking for a Backend Developer to work on our SaaS platform. You will be responsible for developing and maintaining our Django-based backend services, APIs, and database systems. Experience with PostgreSQL, Redis, and AWS is a plus. Remote work with occasional meetings in Makati.",
            "domain": "onlinejobs.ph",
            "salary": "P60,000 - P80,000 per month",
            "job_type": "Full-time",
            "hours_per_week": 40,
            "tags": ["Python", "Django", "Backend", "PostgreSQL", "AWS"],
        },
        {
            "id": "55566677",
            "link": "https://ph.jobstreet.com/job/55566677",
            "title": "Frontend Developer (React/Vue)",
            "company": "Digital Solutions Inc",
            "posted_date": "2026-06-18T15:30:00",
            "details": "We need a Frontend Developer to build modern, responsive web applications for our clients. Proficiency in React or Vue.js is required. You will work closely with our design and backend teams to deliver high-quality user interfaces. Knowledge of TypeScript and modern CSS frameworks is a big plus.",
            "domain": "ph.jobstreet.com",
            "salary": "P45,000 - P65,000 per month",
            "job_type": "Full-time",
            "hours_per_week": 40,
            "tags": ["React", "Vue.js", "Frontend", "TypeScript", "CSS"],
        },
    ]

    with open("jobs.json", "w", encoding="utf-8") as f:
        json.dump(sample_jobs, f, indent=2)

    print(f"Created sample jobs.json with {len(sample_jobs)} job postings")


def demo_queries(rag: JobPostingRAG):
    """Run demonstration queries."""
    print("\n" + "=" * 60)
    print("DEMO QUERIES")
    print("=" * 60)

    queries = [
        {
            "name": "React jobs on OnlineJobs.ph",
            "query": "React developer",
            "filters": {"domain": "onlinejobs.ph"},
        },
        {
            "name": "AI/Automation jobs posted recently",
            "query": "AI automation",
            "filters": {"posted_after": "2026-06-19"},
        },
        {
            "name": "Full-time Python jobs",
            "query": "Python Django",
            "filters": {"job_type": "Full-time", "tags": ["Python"]},
        },
        {
            "name": "High-paying jobs (salary >= $1000 USD)",
            "query": "software engineer",
            "filters": {"salary_min": 1000},
        },
        {"name": "All jobs (no filters)", "query": "developer", "filters": {}},
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: '{q['query']}'")
        if q["filters"]:
            print(f"Filters: {q['filters']}")

        results = rag.semantic_search(query=q["query"], k=5, **q["filters"])

        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result['title']}")
            print(f"     Company: {result.get('company', 'N/A')}")
            print(f"     Domain: {result['domain']}")
            print(f"     Posted: {result['posted_date']}")
            print(f"     Salary: {result.get('salary', 'N/A')}")
            print(f"     Score: {result['score']:.3f}")
            print(f"     Link: {result['link']}")
            if result.get("tags"):
                print(f"     Tags: {', '.join(result['tags'])}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Job Postings RAG Demo")
    print("=" * 60)

    if not os.path.exists("jobs.json"):
        create_sample_jobs_json()

    print("\nInitializing RAG pipeline...")
    rag = JobPostingRAG(
        embedding_model="all-MiniLM-L6-v2",
        vector_store_path="job_postings_index",
        jobs_file="jobs.json",
    )

    rag.initialize(rebuild=True)

    demo_queries(rag)

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (Press Ctrl+C to exit)")
    print("=" * 60)
    print("\nEnter your query (e.g., 'React jobs on onlinejobs.ph'):")

    while True:
        try:
            user_query = input("\n> ").strip()
            if not user_query:
                continue

            query = user_query
            filters = {}

            if " on " in user_query.lower():
                parts = user_query.lower().split(" on ")
                if len(parts) > 1:
                    query = parts[0]
                    domain = parts[1].strip()
                    if domain in ["onlinejobs.ph", "ph.jobstreet.com", "linkedin.com"]:
                        filters["domain"] = domain

            if "full-time" in user_query.lower():
                filters["job_type"] = "Full-time"
            elif "part-time" in user_query.lower():
                filters["job_type"] = "Part-time"

            results = rag.semantic_search(query=query, k=10, **filters)

            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n  {i}. {result['title']}")
                print(f"     Company: {result.get('company', 'N/A')}")
                print(f"     Domain: {result['domain']}")
                print(f"     Posted: {result['posted_date']}")
                print(f"     Score: {result['score']:.3f}")
                print(f"     Link: {result['link']}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
