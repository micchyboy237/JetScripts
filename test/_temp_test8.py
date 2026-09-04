import asyncio
import inspect
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ============================================================================
# 1. OUTPUT DIR SETUP
# ============================================================================
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LightRAG working dir lives INSIDE output dir for containment
WORKING_DIR = str(OUTPUT_DIR / "rag_storage")
os.makedirs(WORKING_DIR, exist_ok=True)

# Rich console initialized early for styled module loading messages
console = Console()

# ============================================================================
# 2. LOCAL MODULE INJECTION (Multi-Repo Support)
# ============================================================================
LOCAL_REPO_MODULES = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/vendors/nano-vectordb",
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/LightRAG",
]

_loaded_modules = []
_missing_modules = []

for _repo_path in LOCAL_REPO_MODULES:
    _path = Path(_repo_path)
    if _path.exists():
        sys.path.insert(0, str(_path))
        _loaded_modules.append(str(_path))
    else:
        _missing_modules.append(str(_path))

if _loaded_modules:
    for _m in _loaded_modules:
        console.print(f"✅ Loaded local module: [cyan]{_m}[/]")
if _missing_modules:
    for _m in _missing_modules:
        console.print(f"⚠️  Local module not found (using installed): [yellow]{_m}[/]")

# ============================================================================
# 3. CONFIG IMPORT & RICH CONSOLE
# ============================================================================
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DIMS,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
)
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import setup_logger, wrap_embedding_func_with_attrs

setup_logger("lightrag", level="INFO")
logger = logging.getLogger("lightrag.stream")

config_summary = Text.from_markup(
    f"[bold cyan]LLM:[/] {LLM_MODEL} @ {LLM_BASE_URL}\n"
    f"[bold cyan]Embed:[/] {EMBED_MODEL} (dim={EMBED_DIMS}) @ {EMBED_BASE_URL}\n"
    f"[bold cyan]WorkDir:[/] {WORKING_DIR}\n"
    f"[bold cyan]Output:[/] {OUTPUT_DIR}"
)
console.print(
    Panel(config_summary, title="⚙️ Active Configuration", border_style="blue")
)


# ============================================================================
# 4. MODEL FUNCTIONS (Config-Driven)
# ============================================================================
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="sk-no-key",
        base_url=LLM_BASE_URL,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_DIMS, max_token_size=8192, model_name=EMBED_MODEL
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed.func(
        texts, model=EMBED_MODEL, api_key="sk-no-key", base_url=EMBED_BASE_URL
    )


# ============================================================================
# 5. SAMPLE DOCUMENT FOR INGESTION
# ============================================================================
# Replace this with real file reading or your own corpus loader.
# LightRAG REQUIRES ainsert() before queries return meaningful results.
SAMPLE_DOCUMENT = """
LightRAG is a lightweight retrieval-augmented generation framework that combines 
knowledge graphs with vector search. It uses a dual-level architecture to manage 
both specific entity relationships and abstract thematic connections across documents.

Key themes in modern RAG systems include:
1. Graph-enhanced retrieval - Using knowledge graphs to capture entity relationships 
   that pure vector similarity misses.
2. Multi-granularity indexing - Chunking documents at multiple levels (entity, relation, 
   chunk) for flexible query resolution.
3. Streaming responses - Delivering partial results incrementally for better UX.
4. Local-first deployment - Running entirely on local hardware without cloud dependencies.
5. Hybrid search modes - Combining naive, local, global, and hybrid strategies to 
   balance precision and recall based on query type.

The framework supports various backends including Neo4j for graph storage, 
nano-vectordb for lightweight vector storage, and OpenAI-compatible APIs for 
LLM and embedding inference.
"""


# ============================================================================
# 6. IMPROVED STREAM HANDLER WITH FILE SAVE
# ============================================================================
async def print_and_save_stream(stream, output_file: Path):
    """Stream to rich console AND accumulate for file save."""
    start = time.perf_counter()
    count = 0
    buffer = []

    console.print("\n[bold green]📝 Streaming Response:[/]\n")

    async for chunk in stream:
        if chunk:
            console.print(chunk, end="", highlight=False)
            buffer.append(chunk)
            count += 1

    total = time.perf_counter() - start
    full_response = "".join(buffer)

    console.print(f"\n\n✅ [bold]{count}[/] chunks in [cyan]{total:.2f}s[/]")

    output_file.write_text(full_response, encoding="utf-8")
    console.print(
        f"💾 Saved output to: [link=file://{output_file}]{output_file}[/link]"
    )

    return full_response


# ============================================================================
# 7. MAIN ENTRY POINT
# ============================================================================
async def main():
    rag = None
    try:
        console.rule("[bold yellow]Initializing LightRAG")
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            llm_model_name=LLM_MODEL,
            embedding_func=embedding_func,
        )
        await rag.initialize_storages()

        # --- Validate embedding function works before ingestion ---
        console.rule("[bold yellow]Validating Embedding Function")
        test_texts = ["Test embedding validation."]
        test_embedding = await rag.embedding_func(test_texts)
        detected_dim = test_embedding.shape[1]
        console.print(
            f"✅ Embedding OK — detected dim: [cyan]{detected_dim}[/] (expected: {EMBED_DIMS})"
        )
        if detected_dim != EMBED_DIMS:
            console.print(
                f"[bold red]❌ DIMENSION MISMATCH! Config says {EMBED_DIMS} but model returns {detected_dim}[/]"
            )
            console.print(
                "[yellow]Update EMBED_DIMS in config or change embedding model.[/]"
            )
            return

        # --- Ingest document(s) BEFORE querying ---
        console.rule("[bold yellow]Ingesting Document")
        await rag.ainsert(SAMPLE_DOCUMENT)
        console.print("✅ Document ingested successfully")

        # --- Query with streaming ---
        query = "Summarize the key themes in this corpus."
        console.rule(f"[bold yellow]Query: {query}")

        resp = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid", stream=True),
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"response_{timestamp}.txt"

        if inspect.isasyncgen(resp):
            await print_and_save_stream(resp, output_file)
        else:
            console.print(str(resp))
            output_file.write_text(str(resp), encoding="utf-8")
            console.print(
                f"💾 Saved output to: [link=file://{output_file}]{output_file}[/link]"
            )

    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/]")
        raise
    finally:
        if rag:
            await rag.finalize_storages()
            console.rule("[bold green]Done")


if __name__ == "__main__":
    asyncio.run(main())
