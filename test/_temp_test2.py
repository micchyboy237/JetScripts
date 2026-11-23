"""
Topic Modeling Pipeline using Turftopic with Local Llama.cpp Integration

This module provides a complete workflow for:
1. Clustering documents into topics using KeyNMF
2. Generating human-readable topic labels using local Llama.cpp LLMs
3. Using local Llama.cpp Embedders for vectorization
4. Saving comprehensive results including keywords, representative documents, and metadata

Usage:
    from topic_modeling_pipeline import generate_topics

    documents = ["comment 1", "comment 2", ...]
    results = generate_topics(documents, n_topics=10)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

# Jet Adapters imports
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
)
from jet.adapters.llama_cpp.embed_utils import embed as jet_embed
from jet.adapters.llama_cpp.factory import get_embedding_client, get_llm_client
from jet.adapters.llama_cpp.llm_utils import chat
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Turftopic imports
from turftopic import KeyNMF
from turftopic.analyzers.base import Analyzer
from turftopic.encoders.base import ExternalEncoder

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LlamaCppEncoder(ExternalEncoder):
    """
    Turftopic-compatible Encoder that uses jet.adapters.llama_cpp.embed_utils.

    This allows KeyNMF to use your local embedding server instead of
    downloading sentence-transformers locally.
    """

    def __init__(
        self,
        model: str = EMBED_MODEL,
        batch_size: int = 64,
        base_url: str = EMBED_BASE_URL,
    ):
        self.model = model
        self.batch_size = batch_size
        self.base_url = base_url
        # Initialize client to ensure connectivity early if desired,
        # though jet_embed handles lazy initialization.
        self._client = get_embedding_client(base_url=base_url)

    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        """
        Encodes sentences into an embedding matrix using jet's optimized embed function.

        Args:
            sentences: Iterable of strings to embed.

        Returns:
            ndarray of shape (n_docs, n_dimensions)
        """
        # Convert iterable to list to allow length check and multiple passes if needed
        sentences_list = list(sentences)

        if not sentences_list:
            return np.array([])

        # Use jet's embed utility which handles batching, deduplication,
        # and progress bars internally.
        # We disable internal progress bar here because we manage it in the pipeline
        embeddings = jet_embed(
            text=sentences_list,
            model=self.model,
            return_format="numpy",
            show_progress=False,
            batch_size=self.batch_size,
        )

        return embeddings


class LlamaCppAnalyzer(Analyzer):
    """
    Turftopic-compatible Analyzer that uses jet.adapters.llama_cpp.llm_utils.

    This allows topic labeling/naming to use your local LLM server.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        context: Optional[str] = None,
        use_summaries: bool = False,
        system_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        namer_prompt: Optional[str] = None,
        description_prompt: Optional[str] = None,
        base_url: str = LLM_BASE_URL,
        temperature: float = 0.2,  # Lower temp for consistent labeling
    ):
        super().__init__()
        self.client = get_llm_client(base_url=base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt or self.system_prompt
        self.summary_prompt = summary_prompt or self.summary_prompt
        self.namer_prompt = namer_prompt or self.namer_prompt
        self.description_prompt = description_prompt or self.description_prompt
        self.use_summaries = use_summaries
        self.context = context
        self.temperature = temperature

    def generate_text(self, prompt: str) -> str:
        """
        Generates response to a given prompt using jet's chat utility.
        """
        try:
            result = chat(
                prompt_or_messages=prompt,
                model=self.model_name,
                client=self.client,
                temperature=self.temperature,
                max_tokens=512,  # Short responses for labels/summaries
                system_message=self.system_prompt,
            )
            return result.content.strip()
        except Exception as e:
            console.print(f"[red]Error generating text via LlamaCpp:[/red] {e}")
            raise


def _validate_documents(documents: list[str]) -> list[str]:
    """Validate and clean input documents."""
    if not documents:
        raise ValueError("Documents list cannot be empty")
    if not isinstance(documents, list):
        raise TypeError(f"Expected list of strings, got {type(documents).__name__}")
    cleaned = [
        doc.strip() for doc in documents if doc and isinstance(doc, str) and doc.strip()
    ]
    if not cleaned:
        raise ValueError("No valid documents found after cleaning")
    console.log(
        f"[green]✓[/green] Validated {len(cleaned)} documents from {len(documents)} inputs"
    )
    return cleaned


def _fit_topic_model(
    documents: list[str],
    n_topics: int = 10,
    encoder: Union[str, ExternalEncoder] = "paraphrase-mpnet-base-v2",
    top_n_keywords: int = 15,
    random_state: int = 42,
) -> KeyNMF:
    """Fit KeyNMF topic model on documents."""

    # Handle string encoder vs Custom Encoder object
    if isinstance(encoder, str):
        encoder_str = encoder
        encoder_obj = encoder  # KeyNMF accepts string paths for default behavior
    else:
        encoder_str = type(encoder).__name__
        encoder_obj = encoder

    console.print(
        Panel.fit(
            f"[bold cyan]Fitting KeyNMF Topic Model[/bold cyan]\n"
            f"Topics: {n_topics} | Encoder: {encoder_str}\n"
            f"Keywords per doc: {top_n_keywords}",
            title="Model Configuration",
            border_style="cyan",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Encoding documents and extracting keywords...", total=None
        )

        model = KeyNMF(
            n_components=n_topics,
            encoder=encoder_obj,
            top_n=top_n_keywords,
            random_state=random_state,
        )

        model.fit(documents)

        progress.update(
            task, description="[green]✓ Model fitted successfully", completed=100
        )
    console.log(f"[green]✓[/green] Discovered {model.n_components} topics")
    return model


def _extract_topic_data(
    model: KeyNMF, documents: list[str], top_docs_per_topic: int = 5
) -> dict:
    """Extract comprehensive topic data including keywords and representative documents."""
    console.print("\n[bold cyan]Extracting topic data...[/bold cyan]")
    topic_keywords = {}
    for topic_id in range(model.n_components):
        keywords_dict = model.get_topic(topic_id)
        topic_keywords[topic_id] = list(keywords_dict.keys())
    console.log(f"[green]✓[/green] Extracted keywords for {len(topic_keywords)} topics")

    doc_topic_matrix = model.transform(documents)
    topic_documents = {}
    for topic_id in range(model.n_components):
        topic_scores = doc_topic_matrix[:, topic_id]
        top_doc_indices = topic_scores.argsort()[-top_docs_per_topic:][::-1]
        topic_documents[topic_id] = [
            {
                "text": documents[idx],
                "score": float(topic_scores[idx]),
                "index": int(idx),
            }
            for idx in top_doc_indices
        ]
    console.log(
        f"[green]✓[/green] Extracted {top_docs_per_topic} representative docs per topic"
    )
    doc_assignments = doc_topic_matrix.argmax(axis=1)
    doc_confidence = doc_topic_matrix.max(axis=1)
    return {
        "keywords": topic_keywords,
        "representative_documents": topic_documents,
        "document_assignments": doc_assignments.tolist(),
        "document_confidence": doc_confidence.tolist(),
    }


def _generate_llm_labels(
    topic_keywords: dict,
    topic_documents: dict,
    analyzer: Analyzer,
    use_summaries: bool = False,
) -> dict:
    """Generate human-readable topic labels and descriptions using LLM."""

    console.print(
        Panel.fit(
            f"[bold magenta]Generating LLM Labels[/bold magenta]\n"
            f"Analyzer: {type(analyzer).__name__}\n"
            f"Summaries: {'Enabled' if use_summaries else 'Disabled'}",
            title="LLM Configuration",
            border_style="magenta",
        )
    )

    keywords_list = [topic_keywords[tid] for tid in sorted(topic_keywords.keys())]
    documents_list = [
        [doc["text"] for doc in topic_documents[tid]]
        for tid in sorted(topic_documents.keys())
    ]

    console.log("[yellow]⚠[/yellow] Calling Local LLM for topic labeling...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[magenta]Generating topic names and descriptions...", total=None
        )

        analysis_results = analyzer.analyze_topics(
            keywords=keywords_list,
            documents=documents_list,
            use_summaries=use_summaries,
        )

        progress.update(task, description="[green]✓ Labels generated", completed=100)

    labels = {
        "topic_names": analysis_results.topic_names,
        "topic_descriptions": analysis_results.topic_descriptions,
    }
    if (
        hasattr(analysis_results, "document_summaries")
        and analysis_results.document_summaries
    ):
        labels["document_summaries"] = analysis_results.document_summaries

    console.log(f"[green]✓[/green] Generated {len(labels['topic_names'])} topic labels")
    return labels


def _save_results(
    documents: list[str],
    topic_data: dict,
    llm_labels: dict,
    model_params: dict,
    output_dir: Path,
) -> dict:
    """Save all results to files and return file paths."""
    console.print("\n[bold cyan]Saving results...[/bold cyan]")
    saved_files = {}

    # 1. Topic Summary CSV
    topic_summary_data = []
    for topic_id in sorted(topic_data["keywords"].keys()):
        topic_summary_data.append(
            {
                "topic_id": topic_id,
                "topic_name": llm_labels["topic_names"][topic_id],
                "topic_description": llm_labels["topic_descriptions"][topic_id],
                "keywords": ", ".join(topic_data["keywords"][topic_id][:10]),
                "num_documents": sum(
                    1 for a in topic_data["document_assignments"] if a == topic_id
                ),
                "avg_confidence": sum(
                    topic_data["document_confidence"][i]
                    for i, a in enumerate(topic_data["document_assignments"])
                    if a == topic_id
                )
                / max(
                    sum(1 for a in topic_data["document_assignments"] if a == topic_id),
                    1,
                ),
            }
        )
    topic_summary_df = pd.DataFrame(topic_summary_data)
    topic_summary_path = output_dir / "topic_summary.csv"
    topic_summary_df.to_csv(topic_summary_path, index=False)
    saved_files["topic_summary"] = topic_summary_path
    console.log(f"[green]✓[/green] Saved topic summary: {topic_summary_path.name}")

    # 2. Detailed Topics JSON
    detailed_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_documents": len(documents),
            "num_topics": len(topic_data["keywords"]),
            "model_params": model_params,
        },
        "topics": {},
    }
    for topic_id in sorted(topic_data["keywords"].keys()):
        detailed_data["topics"][str(topic_id)] = {
            "name": llm_labels["topic_names"][topic_id],
            "description": llm_labels["topic_descriptions"][topic_id],
            "keywords": topic_data["keywords"][topic_id],
            "representative_documents": topic_data["representative_documents"][
                topic_id
            ],
            "assigned_document_indices": [
                i
                for i, a in enumerate(topic_data["document_assignments"])
                if a == topic_id
            ],
        }
    detailed_json_path = output_dir / "detailed_topics.json"
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=2, ensure_ascii=False)
    saved_files["detailed_topics"] = detailed_json_path
    console.log(f"[green]✓[/green] Saved detailed topics: {detailed_json_path.name}")

    # 3. Document Assignments CSV
    assignments_data = []
    for i, doc in enumerate(documents):
        assignments_data.append(
            {
                "document_index": i,
                "assigned_topic": topic_data["document_assignments"][i],
                "topic_name": llm_labels["topic_names"][
                    topic_data["document_assignments"][i]
                ],
                "confidence": topic_data["document_confidence"][i],
                "text": doc[:500],
            }
        )
    assignments_df = pd.DataFrame(assignments_data)
    assignments_path = output_dir / "document_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)
    saved_files["document_assignments"] = assignments_path
    console.log(f"[green]✓[/green] Saved document assignments: {assignments_path.name}")

    # 4. Keywords JSON
    keywords_only = {str(tid): kws for tid, kws in topic_data["keywords"].items()}
    keywords_path = output_dir / "keywords.json"
    with open(keywords_path, "w", encoding="utf-8") as f:
        json.dump(keywords_only, f, indent=2, ensure_ascii=False)
    saved_files["keywords"] = keywords_path
    console.log(f"[green]✓[/green] Saved keywords: {keywords_path.name}")

    # 5. LLM Labels JSON
    labels_data = {
        "topic_names": llm_labels["topic_names"],
        "topic_descriptions": llm_labels["topic_descriptions"],
    }
    if "document_summaries" in llm_labels:
        labels_data["document_summaries"] = llm_labels["document_summaries"]
    labels_path = output_dir / "llm_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)
    saved_files["llm_labels"] = labels_path
    console.log(f"[green]✓[/green] Saved LLM labels: {labels_path.name}")

    return saved_files


def _display_results_table(topic_data: dict, llm_labels: dict) -> None:
    """Display a formatted table of results in console."""
    table = Table(
        title="Topic Modeling Results", show_header=True, header_style="bold cyan"
    )
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Topic Name", style="green")
    table.add_column("Top Keywords", style="yellow")
    table.add_column("Docs", justify="right")
    table.add_column("Description", style="dim")
    for topic_id in sorted(topic_data["keywords"].keys()):
        num_docs = sum(1 for a in topic_data["document_assignments"] if a == topic_id)
        keywords_str = ", ".join(topic_data["keywords"][topic_id][:5])
        table.add_row(
            str(topic_id),
            llm_labels["topic_names"][topic_id],
            keywords_str,
            str(num_docs),
            llm_labels["topic_descriptions"][topic_id][:80] + "...",
        )
    console.print(table)


def generate_topics(
    documents: list[str],
    n_topics: int = 10,
    encoder: Union[str, ExternalEncoder] = "paraphrase-mpnet-base-v2",
    top_n_keywords: int = 15,
    top_docs_per_topic: int = 5,
    analyzer: Optional[Analyzer] = None,
    use_local_llm: bool = True,
    llm_model: str = LLM_MODEL,
    embed_model: str = EMBED_MODEL,
    use_summaries: bool = False,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Complete topic modeling pipeline: cluster documents and generate LLM labels.

    Args:
        documents: List of text documents (e.g., video comments)
        n_topics: Number of topics to discover
        encoder: Sentence transformer model string OR ExternalEncoder instance (e.g. LlamaCppEncoder)
        top_n_keywords: Number of keywords to extract per document
        top_docs_per_topic: Number of representative documents per topic
        analyzer: Custom Analyzer instance. If None, one is created based on use_local_llm.
        use_local_llm: If True and analyzer is None, uses LlamaCppAnalyzer. Else OpenAI.
        llm_model: Model ID for LLM (used if creating LlamaCppAnalyzer)
        embed_model: Model ID for Embeddings (used if creating LlamaCppEncoder and encoder is str)
        use_summaries: Whether to summarize documents before labeling
        random_state: Random seed for reproducibility
        output_dir: Custom output directory

    Returns:
        Dictionary containing results and saved file paths
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup Analyzer if not provided
    if analyzer is None:
        if use_local_llm:
            console.print("[dim]Using Local Llama.cpp Analyzer[/dim]")
            analyzer = LlamaCppAnalyzer(model_name=llm_model)
        else:
            console.print("[dim]Using Default OpenAI Analyzer[/dim]")
            from turftopic.analyzers import OpenAIAnalyzer

            analyzer = OpenAIAnalyzer(model_name="gpt-4o-mini")

    # Setup Encoder if it's a string and we want local
    # Note: If user passed a custom Encoder instance, we respect it.
    # If they passed a string like "paraphrase-mpnet-base-v2", KeyNMF handles it normally.
    # If they want local embeddings, they should pass LlamaCppEncoder() explicitly
    # OR we can intercept here if we want to force local when use_local_llm is true?
    # Let's keep it explicit for safety, but provide a helper hint in docstring.

    console.print(
        Panel.fit(
            "[bold white on blue] Topic Modeling Pipeline [/bold white on blue]\n"
            f"Documents: {len(documents)} | Topics: {n_topics}",
            title="Starting Pipeline",
            border_style="blue",
        )
    )

    try:
        documents = _validate_documents(documents)

        model = _fit_topic_model(
            documents=documents,
            n_topics=n_topics,
            encoder=encoder,
            top_n_keywords=top_n_keywords,
            random_state=random_state,
        )

        topic_data = _extract_topic_data(
            model=model, documents=documents, top_docs_per_topic=top_docs_per_topic
        )

        llm_labels = _generate_llm_labels(
            topic_keywords=topic_data["keywords"],
            topic_documents=topic_data["representative_documents"],
            analyzer=analyzer,
            use_summaries=use_summaries,
        )

        model_params = {
            "n_topics": n_topics,
            "encoder_type": type(encoder).__name__
            if not isinstance(encoder, str)
            else encoder,
            "top_n_keywords": top_n_keywords,
            "llm_analyzer": type(analyzer).__name__,
            "random_state": random_state,
        }

        saved_files = _save_results(
            documents=documents,
            topic_data=topic_data,
            llm_labels=llm_labels,
            model_params=model_params,
            output_dir=out_dir,
        )

        _display_results_table(topic_data, llm_labels)

        console.print("\n" + "=" * 80)
        console.print("[bold green]✓ Pipeline Complete! Saved Files:[/bold green]\n")
        for file_type, file_path in saved_files.items():
            relative_path = file_path.relative_to(Path.cwd())
            console.print(
                f"  • [link=file://{file_path.resolve()}]{file_path.name}[/link]"
            )
        console.print("\n" + "=" * 80)

        topic_summary_df = pd.read_csv(saved_files["topic_summary"])
        with open(saved_files["detailed_topics"], "r", encoding="utf-8") as f:
            detailed_topics = json.load(f)

        return {
            "topic_summary": topic_summary_df,
            "detailed_topics": detailed_topics,
            "document_assignments": topic_data["document_assignments"],
            "document_confidence": topic_data["document_confidence"],
            "saved_files": {k: str(v) for k, v in saved_files.items()},
            "output_dir": str(out_dir),
        }

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        raise


if __name__ == "__main__":
    sample_comments = [
        "The video quality is amazing, love the 4K resolution!",
        "Audio is too quiet, can barely hear the speaker",
        "Great tutorial, very helpful for beginners",
        "Please make more videos like this one",
        "The background music is distracting",
        "Excellent explanation of the topic",
        "Video keeps buffering, terrible experience",
        "Love your content, been watching for years",
        "Can you add subtitles? Would help a lot",
        "The lighting in this video is poor",
        "Best channel on YouTube, keep it up!",
        "Timestamps would be really useful",
        "Sound quality needs improvement",
        "Very informative and well-researched",
        "Why is the video so long? Could be shorter",
        "Amazing editing and production quality",
        "Please cover more advanced topics",
        "The intro is too long, get to the point",
        "Subtitles are inaccurate, please fix",
        "Outstanding work as always!",
    ]

    # Example 1: Using Local Llama.cpp for BOTH Embeddings and LLM Labeling
    console.print("[bold]Running with Local Llama.cpp Stack[/bold]")

    local_encoder = LlamaCppEncoder(model=EMBED_MODEL)
    local_analyzer = LlamaCppAnalyzer(model_name=LLM_MODEL)

    results = generate_topics(
        documents=sample_comments,
        n_topics=5,
        encoder=local_encoder,  # Use local embedder
        analyzer=local_analyzer,  # Use local LLM
        top_docs_per_topic=3,
    )

    console.print(f"\n[bold]Output directory:[/bold] {results['output_dir']}")
