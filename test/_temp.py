#!/usr/bin/env python3
"""
Robust & debug-friendly inspector for Hugging Face Parquet files
Fixed: pyarrow Table.take() now receives proper Array of indices
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ────────────────────────────────── Debug Logging Setup ──────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

console = Console()


def load_parquet_metadata(file_path: Path) -> dict[str, Any]:
    """Load metadata + safe preview using pyarrow (with debug output)."""
    log.debug(f"Opening Parquet file: {file_path}")
    if not file_path.exists():
        console.print(f"[red]File not found:[/red] {file_path}")
        sys.exit(1)

    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow

    row_count = metadata.num_rows
    log.debug(f"Rows in file: {row_count:,}")
    log.debug(f"Columns detected: {schema.names}")

    # ──────── Safe preview (max 8 rows) ────────
    preview_rows = min(8, row_count)
    log.debug(f"Reading preview of {preview_rows} rows")

    if row_count == 0:
        sample_df = pd.DataFrame({col: pd.Series(dtype=schema.field(col).type.to_pandas_dtype())
                                 for col in schema.names})
        log.warning("File has 0 rows → returning empty preview DataFrame")
    else:
        # Fixed: convert range → pyarrow int64 array
        indices = pa.array(range(preview_rows), type=pa.int64())
        log.debug(f"Created pyarrow indices array: {indices.to_pylist()}")
        table = parquet_file.read().take(indices)
        sample_df = table.to_pandas()
        log.debug(f"Preview table shape: {sample_df.shape}")

    # Heuristics for ReazonSpeech / common HF ASR datasets
    text_col = next((c for c in schema.names if c in {"text", "sentence", "transcription"}), None)
    audio_col = next((c for c in schema.names if c in {"audio", "bytes"}), "audio")
    path_col = next((c for c in schema.names if "path" in c.lower()), None)

    info = {
        "path": file_path,
        "row_count": row_count,
        "column_count": len(schema.names),
        "columns": schema.names,
        "dtypes": {col: str(schema.field(col).type) for col in schema.names},
        "sample": sample_df,
        "text_col": text_col,
        "audio_col": audio_col,
        "path_col": path_col,
        "file_size_gb": file_path.stat().st_size / 1e9,
    }
    log.debug("Metadata dict prepared successfully")
    return info


def display_beautiful_info(info: dict[str, Any]) -> None:
    console.rule("[bold blue]Hugging Face Parquet Dataset Inspector[/bold blue]")

    # Summary
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim")
    table.add_column("Value")
    table.add_row("File", info["path"].name)
    table.add_row("Rows", f"[green]{info['row_count']:,}[/green]")
    table.add_row("Columns", f"[blue]{info['column_count']}[/blue]")
    table.add_row("Size", f"[yellow]{info['file_size_gb']:.3f} GB[/yellow]")
    console.print(table)

    # Columns table
    col_table = Table(title="Columns & Types", box=box.SIMPLE_HEAVY)
    col_table.add_column("Column", style="cyan")
    col_table.add_column("PyArrow Type", style="green")
    col_table.add_column("Preview", style="white")

    for col in info["columns"]:
        val = info["sample"][col].iloc[0] if info["row_count"] > 0 else "<empty>"
        if col == info["audio_col"] and isinstance(val, dict):
            preview = f"bytes: {len(val.get('bytes', b'')):,} | rate: {val.get('sampling_rate') or val.get('sample_rate')}"
        elif col == info["path_col"]:
            preview = str(val)[:70] + ("..." if len(str(val)) > 70 else "")
        elif pd.isna(val):
            preview = "<null>"
        else:
            preview = str(val)[:80] + ("..." if len(str(val)) > 80 else "")
        col_table.add_row(col, info["dtypes"][col], preview)

    console.print(col_table)

    # Text samples
    if info["text_col"] and info["row_count"] > 0:
        preview = info["sample"][[info["text_col"]]].head(8)
        console.print(
            Panel(
                preview.to_string(index=False),
                title=f"[bold green]Sample Transcriptions[/bold green] → {info['text_col']}",
                border_style="bright_blue",
            )
        )

    console.print("\n[dim]Tip: pd.read_parquet(path, columns=['audio', 'text']) for streaming[/dim]")


def main(path_str: str | None = None) -> None:
    default_path = Path("/Users/jethroestrada/.cache/huggingface/hub/datasets--japanese-asr--ja_asr.reazon_speech_all/snapshots/10c81088a41b64a99a94f5847a437e248b6a963b/subset_0/train-00000-of-00261.parquet")
    path = Path(path_str or default_path).expanduser().resolve()

    console.print(f"[bold]Inspecting:[/bold] {path.name}")
    log.debug(f"Resolved path: {path}")

    info = load_parquet_metadata(path)
    display_beautiful_info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beautiful Parquet inspector (debug mode)")
    parser.add_argument("file", nargs="?", help="Path to .parquet file")
    parser.add_argument("--quiet", action="store_true", help="Suppress debug logs")
    args = parser.parse_args()

    if args.quiet:
        log.setLevel(logging.INFO)

    main(args.file)