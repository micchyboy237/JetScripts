Here are the workflow diagrams and full Python implementations for each scenario. These examples use `langchain` and `tiktoken` for token management, but the architectural patterns are framework-agnostic.

### 1. Markdown Repo: Structure-Aware Map-Reduce

**Workflow Diagram:**
```mermaid
graph TD
    A[Scan docs/ Folder] --> B{Parse Frontmatter}
    B -- Irrelevant --> C[Skip File]
    B -- Relevant --> D[Split by H1/H2 Headers]
    D --> E[Inject Parent Headers into Each Chunk]
    E --> F[Map: Summarize Each Chunk]
    F --> G{Total Summary Tokens > Budget?}
    G -- Yes --> H[Recursive Reduce: Summarize Summaries]
    G -- No --> I[Final Reduce: Synthesize Repo Summary]
    H --> I
```

**Full Implementation:**
```python
import os, re, tiktoken
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Small context model
enc = tiktoken.encoding_for_model("gpt-4o-mini")
TOKEN_BUDGET = 3500

def load_and_filter_markdown(docs_dir: str, filter_tag: str = "core"):
    """Filter files using frontmatter BEFORE consuming context."""
    relevant_files = []
    for root, _, files in os.walk(docs_dir):
        for f in files:
            if not f.endswith(".md"): continue
            path = os.path.join(root, f)
            content = open(path).read()
            # Simple frontmatter parse (use python-frontmatter in production)
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match and filter_tag in match.group(1):
                relevant_files.append(content)
    return relevant_files

def structure_aware_chunk(markdown_docs: list[str]):
    """Split by headers, preserving hierarchy in metadata."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "H1"), ("##", "H2"), ("###", "H3")
    ])
    all_chunks = []
    for doc in markdown_docs:
        chunks = splitter.split_text(doc)
        for chunk in chunks:
            # Re-inject parent headers so small model has context
            header_ctx = " > ".join(
                v for k, v in chunk.metadata.items() if v
            )
            chunk.page_content = f"[Section: {header_ctx}]\n{chunk.page_content}"
            all_chunks.append(chunk)
    return all_chunks

async def map_reduce_summarize(chunks, budget=TOKEN_BUDGET):
    map_prompt = ChatPromptTemplate.from_template(
        "Summarize this documentation section in ≤150 tokens. "
        "Focus on API contracts, constraints, and key decisions.\n\n{text}"
    )
    reduce_prompt = ChatPromptTemplate.from_template(
        "Synthesize these section summaries into a unified repo overview. "
        "Eliminate redundancy. Max {max_tokens} tokens.\n\n{summaries}"
    )

    # MAP phase
    summaries = []
    for chunk in chunks:
        resp = await llm.ainvoke(map_prompt.format_messages(text=chunk.page_content))
        summaries.append(resp.content)

    # RECURSIVE REDUCE phase
    while len(enc.encode("\n".join(summaries))) > budget:
        batch_size = max(2, budget // 200)  # Dynamic batching based on budget
        new_summaries = []
        for i in range(0, len(summaries), batch_size):
            batch = "\n---\n".join(summaries[i:i+batch_size])
            resp = await llm.ainvoke(reduce_prompt.format_messages(
                summaries=batch, max_tokens=budget // 2
            ))
            new_summaries.append(resp.content)
        summaries = new_summaries

    # FINAL REDUCE
    final = await llm.ainvoke(reduce_prompt.format_messages(
        summaries="\n---\n".join(summaries), max_tokens=budget
    ))
    return final.content
```

---

### 2. Docs Site: Hub-First Deduplicated Traversal

**Workflow Diagram:**
```mermaid
graph TD
    A[Crawl Sitemap] --> B[Build Page Graph]
    B --> C[Identify Hub Pages via Link Count]
    C --> D[Extract Main Content Only]
    D --> E[Summarize Hub Pages First]
    E --> F{Hub Summary Indicates Relevance?}
    F -- Yes --> G[Fetch & Summarize Child Pages]
    F -- No --> H[Skip Subtree]
    G --> I[Deduplicate Across Page Summaries]
    I --> J[Link-Aware Final Synthesis]
```

**Full Implementation:**
```python
import trafilatura
from collections import defaultdict

async def summarize_docs_site(sitemap_urls: list[str], llm, budget=3500):
    # Step 1: Build page graph and identify hubs
    page_graph = defaultdict(list)
    page_contents = {}
    for url in sitemap_urls:
        downloaded = trafilatura.fetch_url(url)
        main_content = trafilatura.extract(downloaded, include_tables=True)
        if not main_content: continue
        page_contents[url] = main_content
        # Extract internal links to build graph
        links = trafilatura.extract(downloaded, include_links=True) or ""
        for link_url in sitemap_urls:
            if link_url != url and link_url in links:
                page_graph[url].append(link_url)

    # Step 2: Sort by hub score (outgoing links = importance proxy)
    hub_sorted = sorted(page_contents.keys(),
                        key=lambda u: len(page_graph.get(u, [])),
                        reverse=True)

    # Step 3: Progressive disclosure summarization
    page_summaries = {}
    child_queue = []

    hub_prompt = ChatPromptTemplate.from_template(
        "Summarize this documentation page in ≤200 tokens. "
        "End with: 'RELEVANT_CHILDREN: [list child topics if this page references them]'\n\n{text}"
    )

    for url in hub_sorted[:10]:  # Process top 10 hubs first
        resp = await llm.ainvoke(hub_prompt.format_messages(text=page_contents[url]))
        page_summaries[url] = resp.content
        # Parse relevant children from structured output
        if "RELEVANT_CHILDREN:" in resp.content:
            child_queue.extend(page_graph.get(url, []))

    # Step 4: Summarize only relevant children
    child_prompt = ChatPromptTemplate.from_template(
        "Parent page summary: {parent_summary}\n\n"
        "Summarize this child page in ≤150 tokens, focusing ONLY on details "
        "not already covered in the parent.\n\n{text}"
    )
    for child_url in set(child_queue):
        if child_url in page_summaries or child_url not in page_contents:
            continue
        parent = next((s for u, s in page_summaries.items()
                       if child_url in page_graph.get(u, [])), "N/A")
        resp = await llm.ainvoke(child_prompt.format_messages(
            parent_summary=parent, text=page_contents[child_url]
        ))
        page_summaries[child_url] = resp.content

    # Step 5: Link-aware final synthesis
    combined = "\n\n".join(
        f"[Page: {url}]\n{summary}"
        for url, summary in page_summaries.items()
    )
    final_prompt = ChatPromptTemplate.from_template(
        "Synthesize these documentation page summaries into a coherent guide. "
        "Preserve cross-references between pages. Max {budget} tokens.\n\n{content}"
    )
    final = await llm.ainvoke(final_prompt.format_messages(
        content=combined, budget=budget
    ))
    return final.content
```

---

### 3. RAG Results: Rerank → Dedup → Grounded Synthesis

**Workflow Diagram:**
```mermaid
graph TD
    A[Raw Retrieval Top-50] --> B[Cross-Encoder Reranker]
    B --> C[Top-20 Reranked Chunks]
    C --> D[Embed All Chunks]
    D --> E[Semantic Clustering threshold=0.92]
    E --> F[Select Centroid Per Cluster]
    F --> G{Fit Within Token Budget?}
    G -- No --> H[Drop Lowest-Ranked Clusters]
    G -- Yes --> I[Order: High-Rank at Start & End]
    I --> J[Citation-Grounded Summarization]
    J --> K[Verify Citations Exist in Source]
```

**Full Implementation:**
```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer, CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

async def summarize_rag_results(query: str, raw_chunks: list[dict], llm, budget=3500):
    """
    raw_chunks: [{"id": "c1", "text": "...", "score": 0.8}, ...]
    """
    # Step 1: Rerank
    pairs = [[query, c["text"]] for c in raw_chunks]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(raw_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
    top_k = [item[0] for item in ranked[:20]]

    # Step 2: Semantic deduplication
    embeddings = embedder.encode([c["text"] for c in top_k])
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.08, metric="cosine", linkage="average"
    ).fit(embeddings)

    # Select highest-ranked chunk per cluster
    seen_clusters = set()
    deduped = []
    for chunk, _ in ranked[:20]:
        idx = top_k.index(chunk)
        cluster_id = clustering.labels_[idx]
        if cluster_id not in seen_clusters:
            deduped.append(chunk)
            seen_clusters.add(cluster_id)

    # Step 3: Budget fitting + Lost-in-the-Middle ordering
    final_chunks = []
    used_tokens = 0
    high_rank, low_rank = [], []
    for i, chunk in enumerate(deduped):
        tokens = len(enc.encode(chunk["text"]))
        if used_tokens + tokens > budget - 500:  # Reserve for prompt + output
            break
        used_tokens += tokens
        if i < len(deduped) // 3 or i > 2 * len(deduped) // 3:
            high_rank.append(chunk)
        else:
            low_rank.append(chunk)

    ordered = high_rank[:len(high_rank)//2] + low_rank + high_rank[len(high_rank)//2:]

    # Step 4: Citation-grounded synthesis
    numbered = "\n\n".join(
        f"[{i}] (ID:{c['id']}) {c['text']}" for i, c in enumerate(ordered)
    )
    prompt = ChatPromptTemplate.from_template(
        "Answer the query using ONLY the provided chunks. "
        "Cite every claim as [N]. If information conflicts, note both views.\n\n"
        "Query: {query}\n\nChunks:\n{chunks}"
    )
    response = await llm.ainvoke(prompt.format_messages(query=query, chunks=numbered))

    # Step 5: Post-hoc citation verification
    cited_ids = set(re.findall(r"\[(\d+)\]", response.content))
    valid_indices = {str(i) for i in range(len(ordered))}
    if not cited_ids.issubset(valid_indices):
        response = await llm.ainvoke(
            "Your previous response contained invalid citations. "
            f"Valid indices: {valid_indices}. Regenerate.\n\n" + 
            prompt.format_messages(query=query, chunks=numbered)[0].content
        )

    return response.content
```

---

### 4. Very Long Text: Recursive Abstractive + Running Summary

**Workflow Diagram:**
```mermaid
graph TD
    A[Full Text] --> B[Lightweight Key-Phrase Extraction]
    B --> C[Anchor-Based Intelligent Chunking]
    C --> D[Initialize Running Summary State]
    D --> E{More Chunks?}
    E -- Yes --> F[Update Running Summary with Next Chunk]
    F --> G{Running Summary Exceeds Budget?}
    G -- Yes --> H[Compress Running Summary In-Place]
    G -- No --> E
    H --> E
    E -- No --> I[Final Polish Pass]
    I --> J[Output Summary]
```

**Full Implementation:**
```python
from langchain.text_splitter import TextSplitter
import spacy

nlp = spacy.load("en_core_web_sm")

class AnchorTextSplitter(TextSplitter):
    """Split at paragraph boundaries near key phrase anchors."""
    def split_text(self, text: str) -> list[str]:
        doc = nlp(text[:100000])  # Limit NER pass for speed
        # Find entity positions as anchor points
        anchors = sorted(set(ent.start_char for ent in doc.ents))
        
        chunks, start = [], 0
        target_size = self.chunk_size
        
        for anchor in anchors:
            if anchor - start >= target_size:
                # Find nearest paragraph break before anchor
                search_zone = text[start:anchor]
                last_para = search_zone.rfind("\n\n")
                split_point = start + last_para if last_para > 0 else anchor
                chunks.append(text[start:split_point].strip())
                start = split_point
        
        if start < len(text):
            chunks.append(text[start:].strip())
        return [c for c in chunks if c]

async def summarize_long_text(text: str, llm, budget=3500):
    splitter = AnchorTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    update_prompt = ChatPromptTemplate.from_template(
        "CURRENT RUNNING SUMMARY ({current_tokens} tokens):\n{summary}\n\n"
        "NEW TEXT SEGMENT:\n{new_text}\n\n"
        "Update the running summary with NEW information only. "
        "Do NOT repeat already-covered points. "
        "Maintain chronological/thematic flow. "
        "Output ONLY the updated summary, ≤{max_tokens} tokens."
    )

    compress_prompt = ChatPromptTemplate.from_template(
        "Compress this summary to ≤{target} tokens without losing key facts, "
        "entities, dates, or conclusions. Remove examples and elaborations.\n\n{summary}"
    )

    running_summary = ""
    max_summary_tokens = int(budget * 0.6)  # Leave room for new chunk in prompt

    for chunk in chunks:
        current_tokens = len(enc.encode(running_summary))
        chunk_tokens = len(enc.encode(chunk))

        # Safety: truncate chunk if it alone exceeds remaining budget
        available = budget - current_tokens - 200  # 200 for prompt overhead
        if chunk_tokens > available:
            chunk = enc.decode(enc.encode(chunk)[:available])

        resp = await llm.ainvoke(update_prompt.format_messages(
            summary=running_summary or "(Empty - begin summarizing)",
            current_tokens=current_tokens,
            new_text=chunk,
            max_tokens=max_summary_tokens
        ))
        running_summary = resp.content

        # Compress if running summary grows too large
        if len(enc.encode(running_summary)) > max_summary_tokens:
            comp = await llm.ainvoke(compress_prompt.format_messages(
                summary=running_summary, target=max_summary_tokens
            ))
            running_summary = comp.content

    # Final polish
    polish_prompt = ChatPromptTemplate.from_template(
        "Polish this summary for readability and coherence. "
        "Fix any artifacts from incremental updating. "
        "Max {budget} tokens.\n\n{summary}"
    )
    final = await llm.ainvoke(polish_prompt.format_messages(
        summary=running_summary, budget=budget
    ))
    return final.content
```

---

### Shared Utilities & Dependencies

```bash
pip install langchain langchain-openai tiktoken trafilatura \
            sentence-transformers scikit-learn spacy
python -m spacy download en_core_web_sm
```

### Key Design Decisions Explained

| Decision | Rationale for Small Context |
| :--- | :--- |
| **Frontmatter/Graph pre-filtering** | Eliminates tokens *before* they ever reach the LLM. Cheapest form of compression. |
| **Dynamic batch sizing in recursive reduce** | Fixed batches fail when summary lengths vary. Budget-aware batching guarantees each reduce step fits. |
| **Agglomerative clustering over cosine similarity pairs** | O(n²) pairwise comparison is wasteful. Clustering deduplicates transitively in O(n·k). |
| **Running summary state vs. Map-Reduce for long text** | Map-Reduce loses narrative continuity in plain text. Running summary preserves temporal/causal chains within a bounded window. |
| **Post-hoc citation verification** | Small models hallucinate citations under pressure. A cheap verification pass catches this without expanding context. |
| **SpaCy NER for anchoring, not LLM** | Using the LLM to find split points wastes context on meta-tasks. Lightweight NLP handles structural analysis for free. |

> ⚡ **Production Note:** All async functions above should be wrapped with rate limiting (`asyncio.Semaphore`) and retry logic with exponential backoff. Small-context models often require more API calls due to recursive patterns—budget your rate limits accordingly. Add observability (e.g., LangSmith, Phoenix) to track token usage per pipeline stage and identify where compression ratios degrade.