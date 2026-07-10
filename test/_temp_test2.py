import json
import os
import uuid
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # Or use HuggingFaceEmbeddings for free

# --- 1. MOCK DATA (Your jobs.json structure) ---
JOBS_DATA = [
    {
        "id": "1591976",
        "link": "https://www.onlinejobs.ph/jobseekers/job/1591976",
        "title": "Part-Time Programming Tutor (JavaScript, React, Django) - 5-10 hrs/week",
        "company": "",
        "posted_date": "2026-06-20T19:31:26",
        "details": "I am currently enrolled in an online Computer Science program and am looking for a part-time virtual programming tutor to supplement my coursework. This role is focused on structured learning and code reviews.",
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
        "details": "We're hiring a hands-on builder who can create real systems using AI tools. You'll design and deploy AI agents, automations, and LLM-powered workflows that replace manual processes and support our internal teams. Responsibilities include: 1. Building Python-based agents. 2. Integrating with Slack and Gmail. 3. Monitoring latency and costs. Requirements: Strong Python skills, experience with LangChain or LlamaIndex, and a portfolio of automation projects.",
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
        "details": "About Aphex\nWe're the construction planning platform that's replacing outdated spreadsheets with multiplayer tools that delivery teams love. Major contractors like BAM, Balfour Beatty, SKANSKA, Kier, and others trust us to manage their most complex projects.\n\nThe Role\nAs a Senior Software Engineer, you will lead the development of our core planning engine. You will work with React, Node.js, and PostgreSQL. You must have experience with real-time collaboration tools (WebSockets) and cloud infrastructure (AWS).\n\nBenefits\n- Competitive salary\n- Remote-first culture\n- Health insurance\n- Annual retreats",
        "domain": "linkedin.com",
        "salary": None,
        "job_type": "Full-time",
        "hours_per_week": None,
        "tags": [],
    },
]

# --- 2. CONFIGURATION ---
# Set your OpenAI API Key here or in environment variables
# os.environ["OPENAI_API_KEY"] = "sk-..."
EMBEDDING_MODEL = "text-embedding-3-small"  # Cheap and effective
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
LONG_DETAIL_THRESHOLD = 300  # Characters. If details > this, we split.

# --- 3. PROCESSING LOGIC ---


def format_job_record(job: Dict) -> str:
    """Converts a JSON job object into a structured text block for the 'Parent' chunk."""
    tags_str = ", ".join(job.get("tags", [])) if job.get("tags") else "None"
    return f"""
Job ID: {job["id"]}
Domain: {job["domain"]}
Title: {job["title"]}
Company: {job.get("company", "N/A")}
Salary: {job.get("salary", "N/A")}
Type: {job.get("job_type", "N/A")}
Hours/Week: {job.get("hours_per_week", "N/A")}
Tags: {tags_str}
Posted: {job["posted_date"]}

Description:
{job["details"]}
""".strip()


def process_jobs(jobs: List[Dict]) -> List[Document]:
    """
    Processes jobs into Parent-Child chunks.
    Returns a list of Documents ready for embedding.
    """
    all_documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    for job in jobs:
        parent_id = str(uuid.uuid4())

        # 1. Create the Parent Document (Full Context)
        parent_content = format_job_record(job)
        parent_doc = Document(
            page_content=parent_content,
            metadata={
                "doc_type": "parent",
                "job_id": job["id"],
                "domain": job["domain"],
                "title": job["title"],
                "link": job["link"],
                "parent_id": parent_id,
            },
        )
        all_documents.append(parent_doc)

        # 2. Handle Long Details (Child Chunks)
        details = job["details"]
        if len(details) > LONG_DETAIL_THRESHOLD:
            child_chunks = splitter.split_text(details)
            for i, chunk in enumerate(child_chunks):
                child_doc = Document(
                    page_content=chunk,
                    metadata={
                        "doc_type": "child",
                        "job_id": job["id"],
                        "parent_id": parent_id,  # Link back to parent
                        "chunk_index": i,
                    },
                )
                all_documents.append(child_doc)

    return all_documents


# --- 4. VECTOR STORE SETUP ---


def setup_vector_store(documents: List[Document]):
    """Creates a Chroma DB and embeds the documents."""
    print(f"Embedding {len(documents)} chunks...")

    # Using OpenAI Embeddings. Replace with HuggingFaceEmbeddings() if no API key.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Create Vector Store
    vector_store = Chroma.from_documents(
        documents=documents, embedding=embeddings, collection_name="jobs_rag_demo"
    )
    return vector_store


# --- 5. RETRIEVAL LOGIC (The "RAG" Part) ---


def retrieve_relevant_jobs(query: str, vector_store: Chroma, k=3):
    """
    Searches for relevant chunks.
    If a Child chunk is found, it retrieves the Parent chunk for full context.
    """
    print(f"\n--- Searching for: '{query}' ---")

    # 1. Similarity Search
    results = vector_store.similarity_search_with_score(query, k=k)

    unique_parent_ids = set()
    final_contexts = []

    for doc, score in results:
        meta = doc.metadata
        doc_type = meta.get("doc_type")

        if doc_type == "parent":
            # If we hit the parent directly, use it
            if meta["parent_id"] not in unique_parent_ids:
                final_contexts.append((doc, score))
                unique_parent_ids.add(meta["parent_id"])

        elif doc_type == "child":
            # If we hit a child, we MUST fetch the parent for full context
            parent_id = meta["parent_id"]
            if parent_id not in unique_parent_ids:
                # Fetch the parent document from the store using metadata filter
                parent_docs = vector_store.similarity_search(
                    "",  # Empty query to just fetch by filter
                    k=1,
                    filter={"parent_id": parent_id, "doc_type": "parent"},
                )
                if parent_docs:
                    final_contexts.append((parent_docs[0], score))
                    unique_parent_ids.add(parent_id)

    return final_contexts


# --- 6. MAIN EXECUTION ---

if __name__ == "__main__":
    # Step 1: Process Data
    documents = process_jobs(JOBS_DATA)
    print(f"Generated {len(documents)} total chunks (Parents + Children)")

    # Step 2: Build Vector Store
    # Note: In a real app, you'd check if the collection exists first
    try:
        db = setup_vector_store(documents)
    except Exception as e:
        print(f"Error setting up DB (likely missing API key): {e}")
        print("Please set OPENAI_API_KEY or switch to HuggingFaceEmbeddings")
        exit()

    # Step 3: Test Queries
    queries = [
        "Who is hiring for Python automation?",
        "Find me a part-time JavaScript tutor",
        "What are the benefits at Aphex?",
    ]

    for q in queries:
        contexts = retrieve_relevant_jobs(q, db)
        for doc, score in contexts:
            print(
                f"[Score: {score:.4f}] Job: {doc.metadata['title']} ({doc.metadata['domain']})"
            )
            # Print first 200 chars of context to verify
            print(f"Context Preview: {doc.page_content[:200]}...\n")
