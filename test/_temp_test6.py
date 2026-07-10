"""
09_demo_llamacpp_local_embeddings.py

PURPOSE
-------
Use YOUR local llama.cpp embedding server (running on the Windows box)
as BERTopic's embedding backend, instead of downloading a
sentence-transformers model.

WHY THIS WORKS
--------------
llama-server exposes an OpenAI-compatible `/v1/embeddings` route. BERTopic
doesn't ship a built-in "llama.cpp backend," but it exposes a
`bertopic.backend.BaseEmbedder` class specifically so you can wrap ANY
embedding source - you just implement one method: `embed()`. Internally
we call it through the official `openai` Python SDK, simply pointed at
your local server's base_url. No API key is actually checked by
llama.cpp, but the SDK requires a non-empty string, so we pass a dummy.

YOUR SERVER (per your env vars)
--------------------------------
    export LLAMA_CPP_EMBED_MODEL="nomic-embed-text-v2-moe"
    export LLAMA_CPP_EMBED_DIMS=768
    export HOST_PC="192.168.68.150"
    export LLAMA_CPP_EMBED_URL="http://${HOST_PC}:8081/v1"

Make sure llama-server was started WITH embeddings enabled, e.g.:
    llama-server -m nomic-embed-text-v2-moe.gguf --embeddings \
        --host 0.0.0.0 --port 8081

IMPORTANT MODEL-SPECIFIC NOTE
-------------------------------
nomic-embed-text models (v1.5 and v2-moe) were trained with TASK
PREFIXES and score meaningfully worse without them:
    "search_document: <text>"   for the documents you are clustering
    "search_query: <text>"      only if you later embed a search query
This script prepends "search_document: " automatically - remove/adjust
the `DOC_PREFIX` constant below if you switch to a model that doesn't
use this convention (e.g. plain "all-MiniLM-L6-v2" style models).

TOKEN LIMIT / TRUNCATION
--------------------------
nomic-embed-text-v2-moe has a hard MODEL limit of 512 tokens - this is
the model's own training limit, not just a server flag. Longer inputs
are rejected by llama.cpp's server with a 400 "exceed_context_size_error"
instead of being silently truncated.

Rather than estimate token count from character length, this script asks
the SAME server for an exact count, using its built-in tokenizer
endpoints (these tokenize with the exact model/vocab that's loaded):
  - POST /tokenize    text -> list of token ids
  - POST /detokenize  list of token ids -> text
For each document we tokenize, and only if it exceeds the safe budget do
we slice the token id list and detokenize it back into truncated text
before sending it to /v1/embeddings. This costs one extra HTTP call per
document (two if truncation is needed) but guarantees we never send an
over-length request and never truncate more aggressively than necessary.
If you need the full text of long documents represented well, consider
chunking long documents into multiple passages and averaging/pooling
their embeddings, rather than relying on truncation.

INSTALL (once)
---------------
pip install bertopic scikit-learn openai requests

RUN
---
python 09_demo_llamacpp_local_embeddings.py
"""

import logging
import os
import time

import numpy as np
import requests
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from openai import OpenAI
from sklearn.datasets import fetch_20newsgroups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo09")

# ---------------------------------------------------------------------------
# Config, pulled straight from your shell env vars.
# ---------------------------------------------------------------------------
EMBED_MODEL = os.environ.get("LLAMA_CPP_EMBED_MODEL", "nomic-embed-text-v2-moe")
EMBED_DIMS = int(os.environ.get("LLAMA_CPP_EMBED_DIMS", "768"))
EMBED_URL = os.environ.get("LLAMA_CPP_EMBED_URL", "http://127.0.0.1:8081/v1")

# nomic-embed-* convention - see note above. Set to "" to disable.
DOC_PREFIX = "search_document: "

# llama.cpp servers are usually run with a modest batch size (-ub/-b flags).
# Sending too many strings in one request can hit context-length limits or
# time out, so we chunk requests client-side.
BATCH_SIZE = 32
MAX_RETRIES = 3


# --- Context-length truncation (using the SERVER's real tokenizer) ---------
# nomic-embed-text-v2-moe has a hard MODEL limit of 512 tokens - this is not
# just a server flag (-c), the model itself was only trained up to 512
# tokens. llama.cpp's server rejects over-length inputs with a 400 error
# instead of silently truncating, so we must truncate client-side.
#
# llama-server's /tokenize and /detokenize routes live at the server ROOT,
# not under /v1 (they are llama.cpp-native, not OpenAI-compatible routes).
def _derive_base_server_url(embed_url: str) -> str:
    trimmed = embed_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed.rstrip("/")


BASE_SERVER_URL = _derive_base_server_url(EMBED_URL)
TOKENIZE_URL = f"{BASE_SERVER_URL}/tokenize"
DETOKENIZE_URL = f"{BASE_SERVER_URL}/detokenize"

MAX_MODEL_TOKENS = int(os.environ.get("LLAMA_CPP_CTX_SIZE", "512"))
SAFETY_MARGIN_TOKENS = 16  # headroom for the model's special/BOS tokens
TOKEN_BUDGET = MAX_MODEL_TOKENS - SAFETY_MARGIN_TOKENS


class LlamaCppEmbedder(BaseEmbedder):
    """
    BERTopic-compatible wrapper around a local llama.cpp OpenAI-compatible
    embeddings endpoint. Implements the single method BaseEmbedder requires.
    """

    def __init__(self, client: OpenAI, model: str, dims: int):
        super().__init__()
        self.client = client
        self.model = model
        self.dims = dims

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        all_embeddings: list[list[float]] = []
        n_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

        prepared_all, n_truncated = self._prepare_inputs(documents)
        if n_truncated:
            log.warning(
                "%d/%d documents exceeded the %d-token model budget and were "
                "truncated (via the server's own tokenizer) before embedding.",
                n_truncated,
                len(documents),
                TOKEN_BUDGET,
            )

        for i in range(0, len(prepared_all), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch = prepared_all[i : i + BATCH_SIZE]

            if verbose:
                log.info(
                    "Embedding batch %d/%d (%d texts)...",
                    batch_num,
                    n_batches,
                    len(batch),
                )

            all_embeddings.extend(self._embed_batch_with_retry(batch, batch_num))

        embeddings = np.array(all_embeddings, dtype=np.float32)
        log.info("Embedded %d documents -> shape %s", len(documents), embeddings.shape)

        if embeddings.shape[1] != self.dims:
            log.warning(
                "Embedding dim mismatch: server returned %d dims, "
                "LLAMA_CPP_EMBED_DIMS says %d. Check your model/env var.",
                embeddings.shape[1],
                self.dims,
            )
        return embeddings

    def _prepare_inputs(self, documents: list[str]) -> tuple[list[str], int]:
        """Apply the task prefix, then use the SERVER's real tokenizer to
        check exact token count and truncate anything over budget.
        Returns (prepared_texts, n_truncated)."""
        prepared = []
        n_truncated = 0

        for doc in documents:
            text = f"{DOC_PREFIX}{doc}"
            tokens = self._tokenize(text)

            if len(tokens) > TOKEN_BUDGET:
                truncated_tokens = tokens[:TOKEN_BUDGET]
                text = self._detokenize(truncated_tokens)
                n_truncated += 1

            prepared.append(text)

        return prepared, n_truncated

    @staticmethod
    def _tokenize(text: str) -> list[int]:
        """Exact token count via llama.cpp's own tokenizer (matches the
        loaded model precisely, unlike a character-based estimate)."""
        response = requests.post(TOKENIZE_URL, json={"content": text}, timeout=15)
        response.raise_for_status()
        return response.json()["tokens"]

    @staticmethod
    def _detokenize(tokens: list[int]) -> str:
        response = requests.post(DETOKENIZE_URL, json={"tokens": tokens}, timeout=15)
        response.raise_for_status()
        return response.json()["content"]

    def _embed_batch_with_retry(
        self, batch: list[str], batch_num: int
    ) -> list[list[float]]:
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    encoding_format="float",
                )
                return [item.embedding for item in response.data]
            except Exception as exc:  # noqa: BLE001 - log and retry deliberately
                last_error = exc
                log.warning(
                    "Batch %d failed (attempt %d/%d): %s",
                    batch_num,
                    attempt,
                    MAX_RETRIES,
                    exc,
                )
                time.sleep(1.5 * attempt)  # simple backoff
        raise RuntimeError(
            f"Batch {batch_num} failed after {MAX_RETRIES} attempts"
        ) from last_error


def build_llamacpp_client() -> OpenAI:
    log.info(
        "Connecting to llama.cpp embed server at %s (model=%s, dims=%d)...",
        EMBED_URL,
        EMBED_MODEL,
        EMBED_DIMS,
    )
    # api_key is ignored by llama.cpp unless you started it with --api-key,
    # but the OpenAI SDK requires a non-empty string.
    return OpenAI(base_url=EMBED_URL, api_key="not-needed")


def sanity_check_server(embedder: LlamaCppEmbedder) -> None:
    """Fail fast with a clear message if the server isn't reachable/configured
    for embeddings, rather than surfacing a confusing BERTopic stack trace."""
    log.info("Running a 1-document sanity check against the embed server...")
    try:
        test_vec = embedder.embed(["connectivity check"], verbose=False)
        log.info("Sanity check OK: got vector of shape %s", test_vec.shape)
    except Exception as exc:
        log.error(
            "Could not reach/embed via %s. Confirm llama-server is running "
            "with --embeddings enabled and reachable from this machine.",
            EMBED_URL,
        )
        raise


def load_sample_documents(n_docs: int = 600) -> list[str]:
    # Smaller sample than earlier demos - local CPU/GPU embedding servers
    # are typically slower per-request than a local sentence-transformers
    # call, so we keep this demo quick to run end-to-end.
    log.info("Loading 20-newsgroups sample (n_docs=%d)...", n_docs)
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))[
        "data"
    ]
    docs = [d for d in data if len(d.strip()) > 20][:n_docs]
    log.info("Loaded %d documents.", len(docs))
    return docs


def fit_with_llamacpp_backend(docs: list[str], embedder: LlamaCppEmbedder) -> BERTopic:
    log.info("Fitting BERTopic using the llama.cpp embedding backend...")
    topic_model = BERTopic(embedding_model=embedder, min_topic_size=10, verbose=True)
    topics, _ = topic_model.fit_transform(docs)
    info = topic_model.get_topic_info()
    log.info("Fit complete: %d topics.\n%s", len(info) - 1, info.head(10).to_string())
    return topic_model


if __name__ == "__main__":
    client = build_llamacpp_client()
    embedder = LlamaCppEmbedder(client, model=EMBED_MODEL, dims=EMBED_DIMS)

    sanity_check_server(embedder)

    documents = load_sample_documents()
    model = fit_with_llamacpp_backend(documents, embedder)

    log.info(
        "Done. Reuse `embedder` in any earlier demo file by passing it as "
        "`embedding_model=embedder` to BERTopic(...) instead of a "
        "sentence-transformers model name."
    )
