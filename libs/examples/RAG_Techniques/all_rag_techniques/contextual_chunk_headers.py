# JetScripts/libs/examples/RAG_Techniques/all_rag_techniques/contextual_chunk_headers.py
"""
Contextual Chunk Headers Example with Full Observability.

Demonstrates how adding document-level context (titles) to chunks
improves reranking relevance scores using local jet adapters.
"""

import argparse
import json
import os
import uuid
from typing import List

from jet.adapters.langchain.factory import get_chat_openai
from jet.adapters.llama_cpp.config import (
    LLM_MODEL,
    PHOENIX_REST_API,
    RERANK_BASE_URL,
    RERANK_MODEL,
)
from jet.adapters.llama_cpp.token_utils import count_tokens
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    RerankerAttributes,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import HTTPSpanExporter
from rich.console import Console
from rich.panel import Panel

# -----------------------------------------------------------------------------
# Observability Setup (mirrors react_with_telemetry)
# -----------------------------------------------------------------------------
PROJECT_NAME = "contextual-chunk-headers"
resource = Resource.create({"openinference.project.name": PROJECT_NAME})
provider = TracerProvider(resource=resource)
exporter = HTTPSpanExporter(endpoint=f"{PHOENIX_REST_API}/traces")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
console = Console(force_terminal=True, highlight=False)

# -----------------------------------------------------------------------------
# Prompts & Constants
# -----------------------------------------------------------------------------
DOCUMENT_TITLE_PROMPT = """INSTRUCTIONS
What is the title of the following document?
Your response MUST be ONLY the short title (under 20 words). DO NOT include any explanation, preamble, or document content.
{document_title_guidance}
{truncation_message}

DOCUMENT
{document_text}""".strip()

TRUNCATION_MESSAGE = """Also note that the document text provided below is just the first ~{num_words} words of the document. That should be plenty for this task. Your response should still pertain to the entire document, not just the text provided below.""".strip()

MAX_CONTENT_TOKENS = 4000
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_INDEX = 86
DEFAULT_QUERY = "Nike climate change impact"


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------
def split_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split text into chunks with telemetry."""
    with tracer.start_as_current_span(
        "chunking.split",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "chunking.chunk_size": chunk_size,
            "chunking.input_length": len(text),
        },
    ) as span:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0, length_function=len
        )
        documents = splitter.create_documents([text])
        chunks = [doc.page_content for doc in documents]

        span.set_attribute("chunking.output_chunk_count", len(chunks))
        console.print(f"[green]✅ Split document into {len(chunks)} chunks[/green]")
        return chunks


def truncate_content(
    content: str, max_tokens: int, model: str = LLM_MODEL
) -> tuple[str, int]:
    """Truncate content to max tokens using jet token utils."""
    tokens = count_tokens(content, model=model, prevent_total=False)
    if isinstance(tokens, list):
        token_count = sum(tokens)
    else:
        token_count = tokens

    if token_count <= max_tokens:
        return content, token_count

    # Simple character-level truncation proportional to token ratio
    char_limit = int(len(content) * (max_tokens / max(token_count, 1)))
    truncated = content[:char_limit]
    return truncated, max_tokens


def get_document_title(document_text: str, document_title_guidance: str = "") -> str:
    """Extract document title using ChatLlamaCpp with streaming and telemetry."""
    with tracer.start_as_current_span(
        "llm.title_extraction",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
            SpanAttributes.LLM_PROVIDER: "llama_cpp",
        },
    ) as span:
        truncated_text, num_tokens = truncate_content(document_text, MAX_CONTENT_TOKENS)
        truncation_message = (
            TRUNCATION_MESSAGE.format(num_words=3000)
            if num_tokens >= MAX_CONTENT_TOKENS
            else ""
        )
        prompt = DOCUMENT_TITLE_PROMPT.format(
            document_title_guidance=document_title_guidance,
            document_text=truncated_text,
            truncation_message=truncation_message,
        )

        span.set_attribute(
            SpanAttributes.LLM_INPUT_MESSAGES,
            json.dumps([{"role": "user", "content": prompt[:2000]}]),
        )
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, num_tokens)

        llm = get_chat_openai(
            verbose=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        console.print("[bold cyan]📄 Extracting Title:[/bold cyan] ", end="")
        collected_content: list[str] = []
        for chunk in llm.stream(prompt):
            token = getattr(chunk, "content", None)
            if token:
                collected_content.append(token)
                print(token, end="", flush=True)

        title = "".join(collected_content).strip()
        print(flush=True)

        # Safety: if model regurgitated content, extract first meaningful line
        if len(title) > 200:
            lines = [l.strip() for l in title.split("\n") if l.strip()]
            title = lines[0][:100] if lines else "Unknown Document"
            console.print(
                f"[yellow]⚠️ Title truncated from oversized response: {title}[/yellow]"
            )

        span.set_attribute(
            SpanAttributes.LLM_OUTPUT_MESSAGES,
            json.dumps([{"role": "assistant", "content": title}]),
        )
        return title


def rerank_documents(query: str, chunks: List[str]) -> List[float]:
    """Rerank chunks using local reranker with telemetry and error handling."""
    with tracer.start_as_current_span(
        "reranker.score",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RERANKER.value,
            RerankerAttributes.RERANKER_MODEL_NAME: RERANK_MODEL,
            RerankerAttributes.RERANKER_QUERY: query,
            RerankerAttributes.RERANKER_TOP_K: len(chunks),
        },
    ) as span:
        import requests

        try:
            resp = requests.post(
                f"{RERANK_BASE_URL}/rerank",
                json={
                    "model": RERANK_MODEL,
                    "query": query,
                    "documents": chunks,
                    "top_k": len(chunks),
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            console.print(f"[bold red]❌ Reranker request failed: {e}[/bold red]")
            span.set_attribute("reranker.error", str(e)[:500])
            return [-1.0] * len(chunks)

        results = data.get("results", [])
        if not results:
            console.print("[bold red]❌ Reranker returned empty results[/bold red]")
            span.set_attribute("reranker.error", "empty_results")
            return [-1.0] * len(chunks)

        reranked_indices = [r["index"] for r in results]
        reranked_scores = [r["relevance_score"] for r in results]

        similarity_scores = [0.0] * len(chunks)
        for i, idx in enumerate(reranked_indices):
            if 0 <= idx < len(similarity_scores):
                similarity_scores[idx] = reranked_scores[i]

        for i, score in enumerate(similarity_scores):
            span.set_attribute(
                f"{RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_SCORE}",
                float(score),
            )

        return similarity_scores


def compare_chunk_similarities(
    chunk_index: int, chunks: List[str], document_title: str, query: str
) -> None:
    """Compare similarity scores with and without contextual header."""
    with tracer.start_as_current_span(
        "evaluation.compare_headers",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "evaluation.chunk_index": chunk_index,
            "evaluation.query": query,
            "evaluation.document_title": document_title,
        },
    ) as span:
        if chunk_index >= len(chunks):
            console.print(
                f"[red]❌ Chunk index {chunk_index} out of range (max {len(chunks) - 1})[/red]"
            )
            span.set_attribute("evaluation.error", "index_out_of_range")
            return

        chunk_text = chunks[chunk_index]
        chunk_wo_header = chunk_text
        chunk_w_header = f"Document Title: {document_title}\n\n{chunk_text}"

        scores = rerank_documents(query, [chunk_wo_header, chunk_w_header])
        score_without = scores[0]
        score_with = scores[1]
        delta = score_with - score_without

        span.set_attribute("evaluation.score_without_header", score_without)
        span.set_attribute("evaluation.score_with_header", score_with)
        span.set_attribute("evaluation.score_delta", delta)

        console.print(
            Panel(chunk_text[:500], title=f"Chunk #{chunk_index}", border_style="dim")
        )
        console.print(f"[bold]Query:[/bold] {query}")

        if score_without == -1.0 or score_with == -1.0:
            console.print("[bold red]⚠️ Reranker failed – scores unavailable[/bold red]")
        else:
            console.print(f"[yellow]Score WITHOUT header:[/yellow] {score_without:.4f}")
            console.print(f"[green]Score WITH header:   [/green] {score_with:.4f}")
            color = "bold green" if delta > 0 else "bold red"
            console.print(f"[{color}]Delta: {delta:+.4f}[/{color}]")


def load_document(file_path: str) -> str:
    """Load document or raise FileNotFoundError if missing."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError(f"File not found: {file_path}")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate contextual chunk headers with full observability."
    )
    parser.add_argument(
        "--file",
        "-f",
        default="data/nike_2023_annual_report.txt",
        help="Path to document file (default: data/nike_2023_annual_report.txt)",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-index",
        "-i",
        type=int,
        default=DEFAULT_CHUNK_INDEX,
        help=f"Chunk index to inspect (default: {DEFAULT_CHUNK_INDEX})",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=DEFAULT_QUERY,
        help=f'Search query for comparison (default: "{DEFAULT_QUERY}")',
    )
    args = parser.parse_args()

    session_id = str(uuid.uuid4())
    console.rule("[bold blue]Contextual Chunk Headers Evaluation")

    with tracer.start_as_current_span(
        "contextual_headers.session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            "session.file": args.file,
            "session.chunk_size": args.chunk_size,
            "session.chunk_index": args.chunk_index,
            "session.query": args.query,
        },
    ) as root_span:
        document_text = load_document(args.file)
        chunks = split_into_chunks(document_text, chunk_size=args.chunk_size)
        document_title = get_document_title(document_text)
        compare_chunk_similarities(args.chunk_index, chunks, document_title, args.query)

        root_span.set_attribute("session.total_chunks", len(chunks))
        root_span.set_attribute("session.document_title", document_title)

    # Print trace link
    phoenix_host = PHOENIX_REST_API.rstrip("/")
    if phoenix_host.endswith("/v1"):
        phoenix_host = phoenix_host[:-3]
    trace_id_hex = format(root_span.get_span_context().trace_id, "032x")
    trace_url = f"{phoenix_host}/redirects/traces/{trace_id_hex}"

    console.rule("[bold]🔗 Trace Link")
    console.print(f"[link={trace_url}]{trace_url}[/link]", style="bold cyan")
    console.print(f"[dim]Session ID: {session_id}[/dim]")


if __name__ == "__main__":
    main()
