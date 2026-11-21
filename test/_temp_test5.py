"""
CrewAI with Local llama.cpp: Smart RAG + Web Search/Extraction
Uses your jet.adapters.llama_cpp.config module
"""

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
from pydantic import Field, PrivateAttr

# Add local CrewAI source paths (must come before crewai imports)
crewai_paths = [
    "/Users/jethroestrada/Desktop/External_Projects/AI/agents/crewAI/lib/crewai/src",
    "/Users/jethroestrada/Desktop/External_Projects/AI/agents/crewAI/lib/crewai-core/src",
    "/Users/jethroestrada/Desktop/External_Projects/AI/agents/crewAI/lib/crewai-files/src",
    "/Users/jethroestrada/Desktop/External_Projects/AI/agents/crewAI/lib/crewai-tools/src",
]

for path in crewai_paths:
    if Path(path).exists():
        sys.path.insert(0, str(path))
    else:
        print(f"⚠️  Warning: Local CrewAI path not found: {path}")

# Now the imports below will resolve to the local source
from crewai import LLM, Agent, Crew, Memory, Process, Task
from crewai.tools import BaseTool

# Fixed: FileWriteTool -> FileWriterTool
from crewai_tools import FileWriterTool, ScrapeWebsiteTool, SerperDevTool
from jet.adapters.llama_cpp import config


class LocalEmbeddingTool(BaseTool):
    """Tool for generating embeddings using local llama.cpp embed server"""

    name: str = "Local Embedding Generator"
    description: str = (
        "Generate vector embeddings for text using local nomic-embed model"
    )
    _embed_url: str = PrivateAttr(default=None)
    _embed_model: str = PrivateAttr(default=None)
    _query_prefix: str = PrivateAttr(default=None)
    _doc_prefix: str = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._embed_url = (
            config.EMBED_BASE_URL or f"{config.EMBED_BASE_HOST}/v1"
            if config.EMBED_BASE_HOST
            else None
        )
        self._embed_model = config.EMBED_MODEL
        self._query_prefix = config.EMBED_QUERY_PREFIX
        self._doc_prefix = config.EMBED_DOC_PREFIX

        if not self._embed_url:
            raise ValueError(
                "Embedding URL not configured. Set LLAMA_CPP_EMBED_URL or LLAMA_CPP_EMBED_HOST"
            )

    def _run(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text"""
        prefix = self._query_prefix if is_query else self._doc_prefix
        prefixed_text = f"{prefix}{text}" if prefix else text

        payload = {
            "model": self._embed_model,
            "input": prefixed_text,
            "encoding_format": "float",
        }

        try:
            response = requests.post(
                f"{self._embed_url}/embeddings", json=payload, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            elif "embedding" in result:
                return result["embedding"]
            else:
                raise ValueError(f"Unexpected embedding response format: {result}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self._run(doc, is_query=False) for doc in documents]

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query"""
        return self._run(query, is_query=True)


class LocalRerankerTool(BaseTool):
    """Tool for reranking search results using local llama.cpp rerank server"""

    name: str = "Local Reranker"
    description: str = "Rerank search results by relevance using local bge-rerank model"
    _rerank_url: str = PrivateAttr(default=None)
    _rerank_model: str = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._rerank_url = (
            config.RERANK_BASE_URL or f"{config.RERANK_BASE_HOST}/v1"
            if config.RERANK_BASE_HOST
            else None
        )
        self._rerank_model = config.RERANK_MODEL

        if not self._rerank_url:
            raise ValueError(
                "Reranker URL not configured. Set LLAMA_CPP_RERANK_URL or LLAMA_CPP_RERANK_HOST"
            )
        if not self._rerank_model:
            raise ValueError(
                "Reranker model not configured. Set LLAMA_CPP_RERANK_MODEL"
            )

    def _run(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance"""
        if not documents:
            return []

        texts = [doc.get("content", doc.get("text", "")) for doc in documents]

        payload = {
            "model": self._rerank_model,
            "query": query,
            "documents": texts,
            "top_n": top_k,
        }

        try:
            response = requests.post(
                f"{self._rerank_url}/rerank", json=payload, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            reranked_docs = []
            for item in result.get("results", []):
                idx = item.get("index", item.get("document_index"))
                score = item.get("relevance_score", item.get("score"))
                if idx is not None and idx < len(documents):
                    doc_copy = documents[idx].copy()
                    doc_copy["rerank_score"] = score
                    reranked_docs.append(doc_copy)

            reranked_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return reranked_docs
        except Exception as e:
            print(f"Warning: Reranking failed ({str(e)}), returning original order")
            return documents[:top_k]


class SmartRAGTool(BaseTool):
    """Smart RAG tool that uses local embeddings, simple vector search, and reranking"""

    name: str = "Smart RAG Search"
    description: str = "Search knowledge base using semantic search with local embeddings and reranking"
    embedding_tool: LocalEmbeddingTool = Field(default_factory=LocalEmbeddingTool)
    reranker_tool: LocalRerankerTool = Field(default_factory=LocalRerankerTool)

    # Simple in-memory vector store
    _documents: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _embeddings: List[List[float]] = PrivateAttr(default_factory=list)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the knowledge base"""
        if not documents:
            return

        texts = [doc.get("content", doc.get("text", "")) for doc in documents]
        embeddings = self.embedding_tool.embed_documents(texts)

        current_id_start = len(self._documents)
        for i, doc in enumerate(documents):
            doc["id"] = current_id_start + i
            doc["embedding"] = embeddings[i]
            self._documents.append(doc)
            self._embeddings.append(embeddings[i])

        print(f"✓ Added {len(documents)} documents to knowledge base")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def _run(self, query: str, top_k: int = 5) -> str:
        """Search knowledge base and return relevant context"""
        if not self._documents:
            return "No documents in knowledge base. Please add documents first."

        # Step 1: Embed query
        query_embedding = self.embedding_tool.embed_query(query)

        # Step 2: Semantic search (retrieve top candidates)
        candidates = []
        for i, doc_embedding in enumerate(self._embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            candidates.append(
                {"index": i, "similarity": similarity, **self._documents[i]}
            )

        # Sort by similarity and get top 10 for reranking
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_candidates = candidates[: min(10, len(candidates))]

        # Step 3: Rerank using local reranker
        reranked = self.reranker_tool._run(query, top_candidates, top_k=top_k)

        # Step 4: Format results
        if not reranked:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(reranked, 1):
            content = doc.get("content", doc.get("text", ""))
            source = doc.get("source", "Unknown")
            score = doc.get("rerank_score", doc.get("similarity", 0))
            context_parts.append(
                f"[{i}] Source: {source} (Score: {score:.3f})\n{content}\n"
            )

        return "\n".join(context_parts)


def create_local_llm() -> LLM:
    """Create LLM instance pointing to local llama.cpp server"""
    llm_url = (
        config.LLM_BASE_URL or f"{config.LLM_BASE_HOST}/v1"
        if config.LLM_BASE_HOST
        else None
    )

    if not llm_url:
        raise ValueError(
            "LLM URL not configured. Set LLAMA_CPP_LLM_URL or LLAMA_CPP_LLM_HOST environment variable"
        )

    llm = LLM(
        model=config.LLM_MODEL,
        custom_openai=True,  # Key parameter for non-OpenAI endpoints
        base_url=llm_url,
        api_key="not-needed",  # llama.cpp doesn't require auth
        temperature=0.7,
        max_tokens=4096,
    )

    print(f"✓ LLM configured: {config.LLM_MODEL} at {llm_url}")
    return llm


def initialize_memory(storage_path: str) -> Memory:
    """
    Initialize CrewAI Memory with proper error handling for LanceDB.
    """
    memory_path = Path(storage_path)

    # Aggressive cleanup of any corrupted state
    if memory_path.exists():
        print(f"🧹 Cleaning up existing memory folder: {memory_path}")
        try:
            shutil.rmtree(memory_path)
        except Exception as e:
            print(f"⚠️  Warning: Could not fully clean {memory_path}: {e}")
            import glob

            for item in glob.glob(str(memory_path / "*")):
                try:
                    if Path(item).is_dir():
                        shutil.rmtree(item)
                    else:
                        Path(item).unlink()
                except:
                    pass

    # Ensure parent directory exists
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📦 Initializing fresh Memory at: {memory_path}")

    # Configure embedder for local llama.cpp
    embedder_config = {
        "provider": "openai",
        "config": {
            "model": config.EMBED_MODEL,
            "base_url": config.EMBED_BASE_URL or f"{config.EMBED_BASE_HOST}/v1"
            if config.EMBED_BASE_HOST
            else None,
            "api_key": "not-needed",
        },
    }

    if not embedder_config["config"]["base_url"]:
        raise ValueError(
            "Embedding base URL not configured. Set LLAMA_CPP_EMBED_URL or LLAMA_CPP_EMBED_HOST"
        )

    try:
        memory = Memory(
            storage_path=str(memory_path),
            embedder=embedder_config,
        )
        print("✓ Memory initialized successfully")
        return memory
    except Exception as e:
        print(f"❌ Failed to initialize Memory: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Ensure llama.cpp embedding server is running")
        print("   2. Check that EMBED_MODEL matches your loaded model")
        print("   3. Verify network connectivity to embedding endpoint")
        raise


def create_safe_filename(topic: str) -> str:
    """
    Create a safe, consistent filename from a topic string.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    slug = topic.lower().strip()
    slug = slug.replace(" ", "_").replace("-", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")

    max_topic_length = 100
    if len(slug) > max_topic_length:
        slug = slug[:max_topic_length].rstrip("_")

    filename = f"report_{date_str}_{slug}.md" if slug else f"report_{date_str}.md"

    return filename


def main():
    """Main demonstration"""
    print("=" * 70)
    print("CREWAI WITH LOCAL LLAMA.CPP + SMART RAG DEMO")
    print("=" * 70)

    # Verify configuration
    print("\n📋 Configuration Check:")
    print(f"  LLM Model: {config.LLM_MODEL}")
    print(f"  LLM URL: {config.LLM_BASE_URL or config.LLM_BASE_HOST}")
    print(f"  Embed Model: {config.EMBED_MODEL}")
    print(f"  Embed URL: {config.EMBED_BASE_URL or config.EMBED_BASE_HOST}")
    print(f"  Rerank Model: {config.RERANK_MODEL}")
    print(f"  Rerank URL: {config.RERANK_BASE_URL or config.RERANK_BASE_HOST}")

    # Step 1: Initialize LLM
    print("\n⚙️  Step 1: Initializing local LLM...")
    try:
        local_llm = create_local_llm()
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        print("\n💡 Tip: Ensure llama.cpp server is running with --api-key flag")
        print("   Example: ./server -m models/qwen3.5.gguf --port 8080 --host 0.0.0.0")
        return

    # Step 2: Initialize Tools
    print("\n🛠️  Step 2: Initializing tools...")

    # Web search and extraction tools
    serper_search = SerperDevTool()
    scrape_website = ScrapeWebsiteTool()
    file_writer = FileWriterTool()

    # Smart RAG components
    # Note: We initialize the tool but DO NOT add sample documents
    try:
        embedding_tool = LocalEmbeddingTool()
        reranker_tool = LocalRerankerTool()
        smart_rag = SmartRAGTool(
            embedding_tool=embedding_tool, reranker_tool=reranker_tool
        )

        # REMOVED: Sample documents addition
        # The knowledge base starts empty.
        # You can manually add docs here if needed, or leave it empty for web-only research.

        print("  ✓ Local Embedding Tool initialized")
        print("  ✓ Local Reranker Tool initialized")
        print("  ✓ Smart RAG Tool initialized (Empty Knowledge Base)")

    except Exception as e:
        print(f"  ⚠️  RAG tools initialization failed: {e}")
        print("  Continuing without RAG (web search only)...")
        smart_rag = None

    # Step 3: Create Agents
    print("\n🤖 Step 3: Creating agents with local LLM...")

    # Research Agent with web search
    researcher = Agent(
        role="Web Research Specialist",
        goal="Find accurate and current information from the web",
        backstory="Expert researcher skilled at finding authoritative sources online",
        llm=local_llm,
        tools=[serper_search, scrape_website],
        verbose=True,
        allow_delegation=False,
    )

    # RAG Analyst (if RAG is available)
    rag_analyst = None
    if smart_rag:
        rag_analyst = Agent(
            role="Knowledge Base Analyst",
            goal="Answer questions using internal knowledge base with semantic search",
            backstory="Specialist in retrieving and synthesizing information from structured knowledge",
            llm=local_llm,
            tools=[smart_rag],
            verbose=True,
            allow_delegation=False,
        )

    # Synthesis Agent
    synthesizer = Agent(
        role="Information Synthesizer",
        goal="Combine information from multiple sources into coherent reports",
        backstory="Senior analyst expert at identifying patterns and resolving contradictions",
        llm=local_llm,
        tools=[file_writer],
        verbose=True,
        allow_delegation=True,
    )

    agent_count = 3 if rag_analyst else 2
    print(f"  ✓ Created {agent_count} agents")

    # Step 4: Define Tasks
    print("\n📝 Step 4: Defining tasks...")

    # Task 1: Web Research
    web_research_task = Task(
        description="""
        Research the topic: "{topic}"
        
        Steps:
        1. Use SerperDevTool to find 5-7 recent articles
        2. Use ScrapeWebsiteTool to extract key information from top 3 sources
        3. Identify main arguments, facts, and expert opinions
        
        Focus on: accuracy, recency, and source credibility
        """,
        expected_output="""Structured summary with:
        - Key findings (bullet points)
        - Source URLs with credibility assessment
        - Contradictions or gaps identified""",
        agent=researcher,
    )

    # Task 2: RAG Query (if available)
    tasks = [web_research_task]

    if rag_analyst:
        rag_task = Task(
            description="""
            Search the knowledge base for information about: "{topic}"
            
            Use the Smart RAG Tool to find relevant documents.
            If the knowledge base is empty, report that no internal data is available.
            """,
            expected_output="Summary of relevant internal knowledge OR confirmation that KB is empty",
            agent=rag_analyst,
        )
        tasks.insert(1, rag_task)

    # Task 3: Synthesis
    synthesis_task = Task(
        description="""
        Synthesize all gathered information into a final report.
        
        Sources to consider:
        - Web research findings: {web_results}
        {rag_context}
        
        Requirements:
        1. Merge insights from all sources
        2. Resolve contradictions with evidence
        3. Provide balanced perspective
        4. Include proper citations
        5. Save report using FileWriterTool with the EXACT filename provided in inputs as 'safe_filename'.
           Do NOT generate your own filename. Use the value from {safe_filename}.
        """,
        expected_output="""Comprehensive markdown report including:
        - Executive Summary
        - Detailed Findings
        - Comparative Analysis
        - Conclusions
        - Bibliography
        - Confirmation of file saved at: {safe_filename}""",
        agent=synthesizer,
        context=tasks[:-1],
    )

    tasks.append(synthesis_task)

    # Step 5: Create and Execute Crew
    print("\n🚀 Step 5: Executing research crew...")

    # --- FIX FOR LANCEDB ERROR ---
    memory_path = "./my_crew_memory"

    try:
        memory = initialize_memory(memory_path)
    except Exception as e:
        print(f"\n❌ Critical: Cannot initialize Memory system: {e}")
        print("   Proceeding WITHOUT memory (short-term context only)")
        memory = None
    # -----------------------------

    # Build agent list safely
    agents_list = [researcher, synthesizer]
    if rag_analyst:
        agents_list.insert(1, rag_analyst)

    crew_kwargs = {
        "agents": agents_list,
        "tasks": tasks,
        "process": Process.sequential,
        "verbose": True,
    }

    if memory:
        crew_kwargs["memory"] = memory

    crew = Crew(**crew_kwargs)

    # Get user input
    topic = input("\nEnter research topic: ").strip()
    if not topic:
        topic = "artificial intelligence ethics"

    # Generate safe filename
    safe_filename = create_safe_filename(topic)
    print(f"📄 Report will be saved as: {safe_filename}")

    print(f"\n{'=' * 70}")
    print(f"RESEARCHING: {topic.upper()}")
    print(f"{'=' * 70}\n")

    try:
        result = crew.kickoff(
            inputs={
                "topic": topic,
                "safe_filename": safe_filename,
                "web_results": "",
                "rag_context": "- RAG search: No internal knowledge available (KB Empty)\n"
                if not smart_rag or not smart_rag._documents
                else "",
            }
        )

        print("\n" + "=" * 70)
        print("✅ RESEARCH COMPLETE!")
        print("=" * 70)
        print(f"\n{result.raw}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
