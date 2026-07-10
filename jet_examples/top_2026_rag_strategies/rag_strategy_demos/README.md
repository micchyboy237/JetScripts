# RAG Strategy Demos

Ten standalone scripts, one per strategy, all pure Python standard library —
**no third-party packages, no APIs, no network calls, no API keys.** Every
script runs offline with `python3 0N_demo_*.py`.

| File | Strategy | Use case |
|---|---|---|
| `01_demo_chunking.py` | Structural / semantic / hierarchical chunking | Long structured docs (policies, runbooks) |
| `02_demo_hybrid_search_rrf.py` | Dense (TF-IDF) + BM25 fused with RRF | Search mixing free text with exact IDs/SKUs |
| `03_demo_reranking.py` | Cross-encoder reranking (heuristic stand-in) | Reordering a first-stage shortlist for precision |
| `04_demo_query_transformation.py` | Multi-query expansion + HyDE | Chat assistants with unpredictable phrasing |
| `05_demo_contextual_retrieval.py` | Contextual chunk augmentation + contextual BM25 | Chunks that lose meaning in isolation |
| `06_demo_multi_source_retrieval.py` | Heterogeneous multi-index retrieval + RRF | Knowledge scattered across formats/sources |
| `07_demo_agentic_rag.py` | Query decomposition + iterative retrieve loop | Multi-hop due-diligence questions |
| `08_demo_adaptive_rag.py` | Complexity-based query routing | Mixed simple/complex query traffic |
| `09_demo_graph_rag.py` | Entity graph construction + traversal | Connecting facts scattered across documents |
| `10_demo_evaluation_ragas.py` | RAGAS-style eval metrics | Pre-deployment regression testing |

## Design choices

- **`common.py`** holds shared building blocks: a from-scratch TF-IDF cosine
  index (stand-in for a dense/embedding retriever), a from-scratch BM25
  index, Reciprocal Rank Fusion, doc loading, and a print helper.
- **No LLM/embedding API calls anywhere.** Any step a production system
  would hand to an LLM (contextual chunk summaries, query rewriting, HyDE,
  query decomposition, sufficiency checks, complexity classification,
  entity extraction) is implemented as a small rule-based **STUB**,
  clearly marked with `# STUB:` and a docstring explaining exactly what
  real model call it replaces.
- Each script is independently runnable and only imports from `common.py`.

## Document schema (`docs.json`)

Every script loads its corpus via `load_docs()` in `common.py`, which reads
`docs.json` from the same folder if present:

```json
[
  {
    "id": "doc_001",
    "title": "Q2 2023 10-Q Filing",
    "text": "Full document text ...",
    "source": "sec_filings",
    "type": "policy",
    "metadata": {
      "company": "ACME Corp",
      "date": "2023-06-30",
      "tags": ["finance", "quarterly"]
    }
  }
]
```

- `source` — used by `06_demo_multi_source_retrieval.py` to build per-source indexes.
- `type` — one of `policy | contract | log | table | article` in the samples; used to pick a demo target document.
- `metadata` — free-form; `company` / `parties` / `system` / `sku` / `policy_id` are used by `05` (contextual retrieval) and `09` (graph RAG) as stand-ins for LLM-extracted context/entities.

**No `docs.json` is included.** If it's missing, `common.py` automatically
falls back to a small built-in 8-document sample set (`SAMPLE_DOCS`) covering
SEC filings, an insurance policy, IT runbooks, a vendor contract, and a
product catalog entry — so every script still runs out of the box. Drop your
own `docs.json` matching the schema above into this folder to use real data.

## Running

```bash
cd rag_demos
python3 01_demo_chunking.py
python3 02_demo_hybrid_search_rrf.py
# ...through 10_demo_evaluation_ragas.py
```
