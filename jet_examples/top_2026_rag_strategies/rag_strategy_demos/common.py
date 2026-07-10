"""
common.py - Shared utilities for the RAG strategy demos (01-10).

No third-party APIs, network calls, or ML libraries are used anywhere
in these demos. Everything (retrieval, ranking, and the steps a real
system would hand to an LLM) is implemented with plain Python so the
scripts run offline, deterministically, and without any API keys.

Where a production system would call an LLM or an embedding API
(contextual chunk summaries, query rewriting, HyDE, query
decomposition, entity extraction, complexity classification) we use
small rule-based stand-ins instead. They're marked `# STUB:` in each
file so it's obvious exactly what you'd swap for a real model call.

"Dense" retrieval is approximated with cosine similarity over
from-scratch TF-IDF vectors (TFIDFIndex) rather than a real embedding
model — same retrieval *shape*, zero external dependencies.

--------------------------------------------------------------------
DOCUMENT SCHEMA (docs.json)
--------------------------------------------------------------------
Every demo loads documents via load_docs(), which expects a JSON file
named docs.json in the same directory, shaped as an array of objects:

[
  {
    "id": "doc_001",              // unique string id
    "title": "Q2 2023 10-Q Filing",
    "text": "Full document text ...",
    "source": "sec_filings",      // logical corpus/source name
    "type": "policy | contract | log | table | article",
    "metadata": {                 // free-form, strategy-specific
        "company": "ACME Corp",
        "date": "2023-06-30",
        "tags": ["finance", "quarterly"]
    }
  },
  ...
]

`source` is used by 06_demo_multi_source_retrieval.py to split the
corpus into per-source indexes. `metadata` fields (company, parties,
system, sku, policy_id) are used by 05 (contextual retrieval) and 09
(graph RAG) as stand-ins for LLM-extracted entities/context.

If docs.json isn't found next to the script, a small built-in sample
set (SAMPLE_DOCS below) is used instead so every demo still runs
out of the box.
"""
import json
import math
import os
import re
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(SCRIPT_DIR, "docs.json")

SAMPLE_DOCS = [
    {"id": "doc_001", "title": "ACME Corp Q2 2023 10-Q Filing", "source": "sec_filings", "type": "policy",
     "text": "This filing covers ACME Corp's second quarter of fiscal year 2023. Revenue grew 3% over "
             "the previous quarter. Operating margin improved due to cost controls in the manufacturing "
             "division.",
     "metadata": {"company": "ACME Corp", "date": "2023-06-30"}},
    {"id": "doc_002", "title": "ACME Corp Q1 2023 10-Q Filing", "source": "sec_filings", "type": "policy",
     "text": "This filing covers ACME Corp's first quarter of fiscal year 2023. Revenue was $314 million, "
             "roughly flat versus the prior year quarter.",
     "metadata": {"company": "ACME Corp", "date": "2023-03-31"}},
    {"id": "doc_003", "title": "Home Insurance Policy Booklet", "source": "insurance_docs", "type": "contract",
     "text": "Section 4: Water damage caused by burst pipes is covered up to $50,000 per incident, "
             "provided the pipe was properly maintained. Flood damage is excluded unless a separate "
             "flood rider is purchased.",
     "metadata": {"policy_id": "HI-2291", "tags": ["water damage", "coverage"]}},
    {"id": "doc_004", "title": "Home Insurance Policy Booklet", "source": "insurance_docs", "type": "contract",
     "text": "Section 7: Claims must be filed within 30 days of the incident. Late claims may be denied "
             "unless the delay was caused by circumstances beyond the policyholder's control.",
     "metadata": {"policy_id": "HI-2291", "tags": ["claims", "deadlines"]}},
    {"id": "doc_005", "title": "IT Runbook: Database Failover", "source": "it_runbooks", "type": "log",
     "text": "Step 1: Confirm primary DB is unreachable via health check. Step 2: Promote the standby "
             "replica in the us-east-2 region. Step 3: Update the connection string in the config "
             "service.",
     "metadata": {"system": "orders-db", "severity": "high"}},
    {"id": "doc_006", "title": "IT Runbook: Cache Invalidation", "source": "it_runbooks", "type": "log",
     "text": "To clear a stale cache entry, run the purge command against the CDN edge nodes, then "
             "verify propagation using the status endpoint before closing the incident.",
     "metadata": {"system": "cdn", "severity": "medium"}},
    {"id": "doc_007", "title": "Vendor Contract: CloudStore Inc.", "source": "contracts", "type": "contract",
     "text": "This agreement is between ACME Corp and CloudStore Inc. Termination requires 90 days "
             "written notice. Either party may terminate for cause with 15 days notice if the other "
             "party fails to remedy a material breach.",
     "metadata": {"company": "ACME Corp", "parties": ["ACME Corp", "CloudStore Inc."], "tags": ["termination"]}},
    {"id": "doc_008", "title": "Product Catalog Entry: SKU-4471", "source": "catalog", "type": "table",
     "text": "SKU-4471: Stainless steel water bottle, 750ml, granite-grey finish. MSRP $24.99. "
             "Warehouse stock: 1,204 units.",
     "metadata": {"sku": "SKU-4471"}},
]


def load_docs(path=None):
    """Load the document corpus from docs.json, falling back to the
    small built-in SAMPLE_DOCS if the file isn't present."""
    path = path or DOCS_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[common] {os.path.basename(path)} not found next to this script — "
          f"using built-in SAMPLE_DOCS ({len(SAMPLE_DOCS)} docs) so the demo still runs.\n")
    return SAMPLE_DOCS


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class TFIDFIndex:
    """From-scratch TF-IDF vector index with cosine similarity search.

    Stands in for a dense/embedding retriever without calling any
    external embedding API or model — same interface (search(query) ->
    ranked (doc, score) pairs), zero dependencies.
    """

    def __init__(self, docs, text_fn=lambda d: d["text"]):
        self.docs = docs
        self.text_fn = text_fn
        self.doc_tokens = [tokenize(text_fn(d)) for d in docs]
        df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.doc_vecs = [self._vectorize(toks) for toks in self.doc_tokens]

    def _vectorize(self, tokens):
        if not tokens:
            return {}
        tf = Counter(tokens)
        vec = {t: (c / len(tokens)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _query_vec(self, query):
        tokens = tokenize(query)
        if not tokens:
            return {}
        tf = Counter(tokens)
        vec = {t: (c / len(tokens)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    @staticmethod
    def _cosine(a, b):
        common_terms = set(a) & set(b)
        return sum(a[t] * b[t] for t in common_terms)

    def search(self, query, top_k=5):
        qvec = self._query_vec(query)
        scores = [(self._cosine(qvec, dv), i) for i, dv in enumerate(self.doc_vecs)]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [(self.docs[i], round(s, 4)) for s, i in scores[:top_k] if s > 0]


class BM25Index:
    """From-scratch Okapi BM25 sparse retriever."""

    def __init__(self, docs, text_fn=lambda d: d["text"], k1=1.5, b=0.75):
        self.docs = docs
        self.text_fn = text_fn
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(text_fn(d)) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0
        df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
        self.doc_tf = [Counter(toks) for toks in self.doc_tokens]

    def _score(self, q_tokens, i):
        score = 0.0
        tf = self.doc_tf[i]
        dl = self.doc_len[i]
        for t in q_tokens:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            freq = tf[t]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1)
        return score

    def search(self, query, top_k=5):
        q_tokens = tokenize(query)
        scores = [(self._score(q_tokens, i), i) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [(self.docs[i], round(s, 4)) for s, i in scores[:top_k] if s > 0]


def reciprocal_rank_fusion(rankings, k=60, top_k=5):
    """Fuse multiple ranked result lists by rank position (not raw
    score, since dense/sparse scores aren't on comparable scales).

    rankings: list of ranked lists, each a list of (doc, score),
              best result first.
    Returns: fused list of (doc, fused_score), best first.
    """
    fused = defaultdict(float)
    doc_lookup = {}
    for ranking in rankings:
        for rank, (doc, _score) in enumerate(ranking, start=1):
            fused[doc["id"]] += 1.0 / (k + rank)
            doc_lookup[doc["id"]] = doc
    merged = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [(doc_lookup[doc_id], round(score, 5)) for doc_id, score in merged[:top_k]]


def print_results(title, results):
    print(f"\n--- {title} ---")
    if not results:
        print("(no results)")
        return
    for rank, (doc, score) in enumerate(results, 1):
        snippet = doc["text"][:90].replace("\n", " ")
        print(f"{rank}. [{score:.4f}] {doc['id']} - {doc['title']}: {snippet}...")
