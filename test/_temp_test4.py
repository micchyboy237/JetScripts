"""
RAG Chunking Demo for Jobs Data
Complete working implementation with evaluation metrics
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ============================================
# 1. DATA MODELS
# ============================================


@dataclass
class Job:
    """Job data model"""

    id: str
    link: str
    title: str
    posted_date: str
    details: str
    domain: str
    company: str = ""
    salary: str = ""
    job_type: str = ""
    hours_per_week: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> "Job":
        """Create Job from dictionary with error handling"""
        return cls(
            id=str(data.get("id", "")),
            link=data.get("link", ""),
            title=data.get("title", ""),
            posted_date=data.get("posted_date", ""),
            details=data.get("details", ""),
            domain=data.get("domain", ""),
            company=data.get("company", ""),
            salary=data.get("salary", ""),
            job_type=data.get("job_type", ""),
            hours_per_week=data.get("hours_per_week"),
            tags=data.get("tags", []) or [],
        )

    def days_since_posted(self) -> int:
        """Calculate days since job was posted"""
        try:
            posted = datetime.fromisoformat(self.posted_date.replace("Z", "+00:00"))
            return (datetime.now() - posted).days
        except:
            return 999


@dataclass
class Chunk:
    """RAG Chunk data model"""

    text: str
    job_id: str
    chunk_type: str  # 'metadata' or 'detail'
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """Text to be embedded"""
        return self.text

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "job_id": self.job_id,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


# ============================================
# 2. CHUNKING STRATEGIES
# ============================================


class SemanticChunker:
    """Intelligent semantic chunking based on document structure"""

    def __init__(self, max_tokens: int = 600, min_tokens: int = 100):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Normalize line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [text]

    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simplified)"""
        # Handle common sentence endings
        text = re.sub(r"([.!?])\s+", r"\1|", text)
        sentences = [s.strip() for s in text.split("|") if s.strip()]
        return sentences

    def split_by_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points as separate chunks"""
        # Match common bullet patterns
        bullet_pattern = r"(?:^|\n)\s*[•\-*]\s+(.+?)(?=\n\s*[•\-*]|\n\n|$)"
        bullets = re.findall(bullet_pattern, text, re.DOTALL)
        return [b.strip() for b in bullets] if bullets else []

    def chunk_semantically(self, text: str) -> List[str]:
        """Main chunking logic with semantic boundaries"""
        if not text or len(text) < 200:
            return [text]

        # Try to find natural boundaries
        chunks = []

        # 1. Check for bullet points
        bullets = self.split_by_bullet_points(text)
        if len(bullets) > 1:
            return bullets

        # 2. Split by paragraphs
        paragraphs = self.split_by_paragraphs(text)

        current_chunk = ""
        for para in paragraphs:
            # If para is a heading (short, no period at end)
            is_heading = len(para.split()) < 8 and not para.endswith(".")

            if is_heading and current_chunk:
                # Start new chunk for heading
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                current_chunk = para + "\n"
            else:
                # Try adding to current chunk
                test_chunk = (
                    f"{current_chunk}\n{para}".strip() if current_chunk else para
                )

                # Estimate tokens (rough: 1 token ~ 4 chars)
                if len(test_chunk) / 4 <= self.max_tokens:
                    current_chunk = test_chunk
                else:
                    # Current chunk is full, start new
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def __call__(self, text: str) -> List[str]:
        """Make chunker callable"""
        return self.chunk_semantically(text)


class HybridChunker:
    """
    Hybrid chunking: metadata + semantic details
    This is the main chunking strategy for jobs data
    """

    def __init__(self, max_detail_tokens: int = 600):
        self.semantic_chunker = SemanticChunker(max_tokens=max_detail_tokens)

    def create_metadata_chunk(self, job: Job) -> Chunk:
        """Create a structured metadata chunk"""
        # Build metadata text with all available fields
        metadata_parts = [
            f"Job ID: {job.id}",
            f"Title: {job.title}",
            f"Domain: {job.domain}",
            f"Posted Date: {job.posted_date}",
        ]

        if job.company:
            metadata_parts.append(f"Company: {job.company}")
        if job.salary:
            metadata_parts.append(f"Salary: {job.salary}")
        if job.job_type:
            metadata_parts.append(f"Job Type: {job.job_type}")
        if job.hours_per_week:
            metadata_parts.append(f"Hours per Week: {job.hours_per_week}")
        if job.tags:
            metadata_parts.append(f"Tags: {', '.join(job.tags)}")

        # Add recency info
        days_ago = job.days_since_posted()
        metadata_parts.append(f"Days Since Posted: {days_ago}")

        metadata_text = "\n".join(metadata_parts)

        return Chunk(
            text=metadata_text,
            job_id=job.id,
            chunk_type="metadata",
            chunk_index=0,
            metadata={
                "domain": job.domain,
                "title": job.title,
                "tags": job.tags,
                "recency_days": days_ago,
            },
        )

    def create_detail_chunks(self, job: Job) -> List[Chunk]:
        """Create semantic chunks from job details"""
        if not job.details:
            return []

        # Get semantic chunks
        detail_parts = self.semantic_chunker(job.details)

        chunks = []
        for i, detail in enumerate(detail_parts):
            # Enrich each chunk with context
            enriched_text = f"""
Title: {job.title}
Domain: {job.domain}
{f"Tags: {', '.join(job.tags)}" if job.tags else ""}

{detail}
""".strip()

            chunk = Chunk(
                text=enriched_text,
                job_id=job.id,
                chunk_type="detail",
                chunk_index=i + 1,  # 0 is reserved for metadata
                metadata={
                    "domain": job.domain,
                    "title": job.title,
                    "tags": job.tags,
                    "detail_part": i,
                },
            )
            chunks.append(chunk)

        return chunks

    def chunk_job(self, job: Job) -> List[Chunk]:
        """Main entry point: create all chunks for a job"""
        chunks = []

        # Always include metadata
        chunks.append(self.create_metadata_chunk(job))

        # Add detail chunks
        chunks.extend(self.create_detail_chunks(job))

        return chunks


# ============================================
# 3. RETRIEVAL ENGINE (Mock for Demo)
# ============================================


class SimpleRetriever:
    """
    Simple retrieval engine for demonstration
    In production, replace with vector database (Pinecone, Weaviate, etc.)
    """

    def __init__(self):
        self.chunks: List[Chunk] = []

    def index_chunks(self, chunks: List[Chunk]):
        """Store chunks for retrieval"""
        self.chunks.extend(chunks)

    def search(
        self, query: str, top_k: int = 5, domain_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Simple keyword + metadata search
        In production, use vector similarity
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for chunk in self.chunks:
            # Domain filter
            if domain_filter and chunk.metadata.get("domain") != domain_filter:
                continue

            # Calculate simple relevance score
            text_lower = chunk.text.lower()

            # Exact phrase match (boost)
            phrase_score = 2.0 if query_lower in text_lower else 0.0

            # Word overlap
            chunk_words = set(text_lower.split())
            overlap = len(query_words & chunk_words) / max(len(query_words), 1)

            # Tag boost
            tag_boost = 0.0
            if "tags" in chunk.metadata:
                tag_match = any(
                    tag.lower() in query_lower for tag in chunk.metadata["tags"]
                )
                tag_boost = 0.5 if tag_match else 0.0

            # Recency boost (for metadata chunks only)
            recency_boost = 0.0
            if chunk.chunk_type == "metadata":
                days = chunk.metadata.get("recency_days", 999)
                if days < 7:
                    recency_boost = 0.3
                elif days < 30:
                    recency_boost = 0.1

            # Combined score
            score = overlap + phrase_score + tag_boost + recency_boost

            if score > 0:
                results.append((chunk, score))

        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ============================================
# 4. EVALUATION METRICS
# ============================================


class RAGEvaluator:
    """Evaluate RAG performance"""

    def __init__(self):
        self.results = []

    def evaluate_retrieval(
        self,
        query: str,
        expected_job_ids: List[str],
        retrieved_chunks: List[Tuple[Chunk, float]],
    ) -> Dict:
        """
        Calculate retrieval metrics
        """
        retrieved_ids = [chunk.job_id for chunk, _ in retrieved_chunks]

        # Precision: how many retrieved are relevant
        relevant_retrieved = [jid for jid in retrieved_ids if jid in expected_job_ids]
        precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0

        # Recall: how many relevant were retrieved
        recall = (
            len(relevant_retrieved) / len(expected_job_ids) if expected_job_ids else 0
        )

        # F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Average rank of relevant chunks
        ranks = []
        for i, (chunk, _) in enumerate(retrieved_chunks):
            if chunk.job_id in expected_job_ids:
                ranks.append(i + 1)
        mean_rank = sum(ranks) / len(ranks) if ranks else float("inf")

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_rank": mean_rank,
            "retrieved_count": len(retrieved_ids),
            "relevant_retrieved": len(relevant_retrieved),
        }


# ============================================
# 5. MAIN APPLICATION
# ============================================


class JobRAGSystem:
    """Complete RAG system for job data"""

    def __init__(self):
        self.chunker = HybridChunker(max_detail_tokens=600)
        self.retriever = SimpleRetriever()
        self.evaluator = RAGEvaluator()
        self.jobs: List[Job] = []

    def load_jobs(self, json_data: List[Dict]):
        """Load and process jobs data"""
        self.jobs = [Job.from_dict(j) for j in json_data]
        print(f"✅ Loaded {len(self.jobs)} jobs")

    def index_all_jobs(self):
        """Chunk and index all jobs"""
        total_chunks = 0

        for job in self.jobs:
            chunks = self.chunker.chunk_job(job)
            self.retriever.index_chunks(chunks)
            total_chunks += len(chunks)

        print(f"✅ Indexed {total_chunks} chunks from {len(self.jobs)} jobs")
        print(f"   - Average chunks per job: {total_chunks / len(self.jobs):.1f}")

    def search_jobs(
        self, query: str, domain: Optional[str] = None, top_k: int = 5
    ) -> List[Dict]:
        """Search for relevant jobs"""
        results = self.retriever.search(query, top_k=top_k, domain_filter=domain)

        # Group results by job
        grouped = {}
        for chunk, score in results:
            if chunk.job_id not in grouped:
                grouped[chunk.job_id] = {
                    "job_id": chunk.job_id,
                    "chunks": [],
                    "max_score": score,
                }
            grouped[chunk.job_id]["chunks"].append(
                {
                    "type": chunk.chunk_type,
                    "text": chunk.text[:300] + "..."
                    if len(chunk.text) > 300
                    else chunk.text,
                    "score": score,
                }
            )

        # Convert to list and sort by max score
        return sorted(
            [v for v in grouped.values()], key=lambda x: x["max_score"], reverse=True
        )

    def evaluate_query(
        self, query: str, expected_job_ids: List[str], domain: Optional[str] = None
    ) -> Dict:
        """Evaluate retrieval performance for a query"""
        retrieved = self.retriever.search(query, top_k=10, domain_filter=domain)
        return self.evaluator.evaluate_retrieval(query, expected_job_ids, retrieved)

    def get_stats(self) -> Dict:
        """Get system statistics"""
        total_chunks = len(self.retriever.chunks)
        metadata_chunks = sum(
            1 for c in self.retriever.chunks if c.chunk_type == "metadata"
        )
        detail_chunks = total_chunks - metadata_chunks

        return {
            "total_jobs": len(self.jobs),
            "total_chunks": total_chunks,
            "metadata_chunks": metadata_chunks,
            "detail_chunks": detail_chunks,
            "avg_chunks_per_job": total_chunks / len(self.jobs) if self.jobs else 0,
            "domains": list(set(j.domain for j in self.jobs)),
        }


# ============================================
# 6. DEMO & USAGE
# ============================================


def demo():
    """Full demonstration of the RAG system"""

    print("=" * 60)
    print("🚀 JOB RAG SYSTEM - COMPLETE DEMO")
    print("=" * 60)

    # Sample data (using your provided jobs)
    sample_jobs = [
        {
            "id": "1591976",
            "link": "https://www.onlinejobs.ph/jobseekers/job/1591976",
            "title": "Part-Time Programming Tutor (JavaScript, React, Django) - 5-10 hrs/week",
            "company": "",
            "posted_date": "2026-06-20T19:31:26",
            "details": """I am currently enrolled in an online Computer Science program and am looking for a part-time virtual programming tutor to supplement my coursework.

This role is focused on structured learning and project-based mentoring. You'll help me with:

• JavaScript fundamentals and ES6+ features
• React hooks and state management
• Django REST framework and ORM
• Building full-stack applications

The ideal candidate has strong communication skills and can explain complex concepts simply. We'll meet 2-3 times per week via video call.

Requirements:
- 2+ years of experience with JavaScript/React
- Working knowledge of Python and Django
- Previous tutoring or teaching experience (preferred)
- Patient and encouraging teaching style""",
            "domain": "onlinejobs.ph",
            "salary": "Negotiable",
            "job_type": "Any",
            "hours_per_week": 10,
            "tags": ["Javascript", "React JS", "Python"],
        },
        {
            "id": "92840537",
            "link": "https://ph.jobstreet.com/job/92840537",
            "title": "AI Agents & Automation Engineer",
            "company": "Online Helpers",
            "posted_date": "2026-06-20T13:47:38.547450",
            "details": """We're hiring a hands-on builder who can create real systems using AI tools. You'll design and deploy AI agents, automations, and LLM-powered workflows that replace manual processes and support our growing operations.

Key Responsibilities:
- Build AI agents using LangChain and similar frameworks
- Design automation pipelines for data processing
- Integrate LLMs with existing APIs and tools
- Monitor and optimize agent performance

Technical Requirements:
- Strong Python programming skills
- Experience with LangChain or similar frameworks
- Knowledge of vector databases (Pinecone, Weaviate)
- Understanding of RAG and prompt engineering

This is a fully remote role with flexible hours. You'll work directly with our CTO and have significant autonomy in your work.""",
            "domain": "ph.jobstreet.com",
            "salary": "$600 – $850 per month (USD)",
            "job_type": "Full time",
            "hours_per_week": None,
            "tags": [],
        },
        {
            "id": "4429633048",
            "link": "https://www.linkedin.com/jobs/view/4429633048",
            "title": "Senior Software Engineer (Full Stack)",
            "company": "Aphex",
            "posted_date": "2026-06-19T00:00:00",
            "details": """About Aphex
We're the construction planning platform that's replacing outdated spreadsheets with multiplayer tools that delivery teams love. Major contractors like BAM, Balfour Beatty, SKANSKA, Kier, and Lendlease use us to plan projects worth over $30B.

The Role
We're looking for a Senior Full Stack Engineer who wants to solve complex problems in the construction industry. You'll work on our core planning platform, building features that thousands of construction professionals use daily.

Tech Stack
• Frontend: React, TypeScript, Tailwind
• Backend: Python, Django, PostgreSQL
• Infrastructure: AWS, Docker, Kubernetes

What We're Looking For
- 5+ years of full-stack development experience
- Strong TypeScript and Python skills
- Experience with large-scale web applications
- Good understanding of system architecture and design patterns
- Passion for clean code and engineering excellence

We offer competitive salary, equity, and benefits. Our team is distributed across the UK and Europe, and we work remotely.""",
            "domain": "linkedin.com",
            "salary": None,
            "job_type": "Full-time",
            "hours_per_week": None,
            "tags": [],
        },
    ]

    # 1. Initialize system
    print("\n📦 Initializing RAG System...")
    rag = JobRAGSystem()

    # 2. Load data
    rag.load_jobs(sample_jobs)

    # 3. Index all jobs
    print("\n🔨 Indexing jobs...")
    rag.index_all_jobs()

    # 4. System statistics
    stats = rag.get_stats()
    print("\n📊 System Statistics:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    # 5. Demonstration queries
    print("\n" + "=" * 60)
    print("🔍 SEARCH DEMONSTRATION")
    print("=" * 60)

    test_queries = [
        ("React developer", None),
        ("AI automation engineer", None),
        ("Python full stack", "linkedin.com"),
        ("part time tutor", None),
        ("construction planning", None),
    ]

    for query, domain_filter in test_queries:
        print(
            f"\n📝 Query: '{query}'"
            + (f" (Domain: {domain_filter})" if domain_filter else "")
        )
        print("-" * 40)

        results = rag.search_jobs(query, domain=domain_filter, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                print(
                    f"\n  {i}. Job ID: {result['job_id']} (Score: {result['max_score']:.2f})"
                )
                for chunk in result["chunks"][:2]:  # Show top 2 chunks
                    print(f"     [{chunk['type']}] {chunk['text'][:100]}...")
        else:
            print("  No results found")

    # 6. Evaluation
    print("\n" + "=" * 60)
    print("📈 EVALUATION METRICS")
    print("=" * 60)

    # Define expected results for each query
    expected = {
        "React developer": ["1591976", "4429633048"],
        "AI automation engineer": ["92840537"],
        "Python full stack": ["4429633048"],
        "part time tutor": ["1591976"],
    }

    for query, expected_ids in expected.items():
        print(f"\nQuery: '{query}'")
        eval_result = rag.evaluate_query(query, expected_ids)
        print(f"  Precision: {eval_result['precision']:.2%}")
        print(f"  Recall: {eval_result['recall']:.2%}")
        print(f"  F1 Score: {eval_result['f1']:.2%}")
        print(
            f"  Mean Rank: {eval_result['mean_rank']:.1f}"
            if eval_result["mean_rank"] != float("inf")
            else "  Mean Rank: N/A"
        )

    # 7. Show chunk examples
    print("\n" + "=" * 60)
    print("📄 CHUNK EXAMPLES")
    print("=" * 60)

    sample_job = rag.jobs[0]
    chunks = rag.chunker.chunk_job(sample_job)

    print(f"\nJob: {sample_job.title}")
    print(f"Total Chunks: {len(chunks)}")

    for chunk in chunks[:3]:  # Show first 3 chunks
        print(f"\n[{chunk.chunk_type.upper()}] Chunk {chunk.chunk_index}")
        print(f"Text Preview: {chunk.text[:200]}...")
        print(f"Metadata: {chunk.metadata}")

    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)


# ============================================
# 7. REUSABLE EXPORTS
# ============================================

__all__ = [
    "Job",
    "Chunk",
    "HybridChunker",
    "SemanticChunker",
    "JobRAGSystem",
    "SimpleRetriever",
    "RAGEvaluator",
    "demo",
]

if __name__ == "__main__":
    demo()
