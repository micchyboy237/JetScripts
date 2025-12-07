#!/usr/bin/env python3
"""
Example: Compress recordings from run_record_mic → ./generated/run_audio_compressor/

Default behavior:
→ Takes all audio files from your run_record_mic folder
→ Compresses to FLAC (lossless)
→ Saves everything into ./generated/run_audio_compressor/
→ Keeps originals (safe by default)

Run with:
    python examples/compress_my_recordings.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from jet.audio.audio_compressor import AudioCodec, CompressionConfig, compress_audio
from rich import print as rprint
from rich.panel import Panel

# ─────────────────────────────────────────────────────────────────────────────
# Your corrected default path
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_AUDIO_DIR = Path(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic"
)

# === OUTPUT DIRECTORY SETUP ===
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)  # Clean start every run
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rprint(f"[dim]Output directory ready:[/dim] {OUTPUT_DIR.resolve()}")


def build_config(args: argparse.Namespace) -> CompressionConfig:
    return CompressionConfig(
        codec=AudioCodec(args.codec.lower()),
        opus_bitrate_kbps=args.bitrate,
        compression_level=args.level,
        keep_original=not args.delete,
        overwrite=args.overwrite,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compress recordings → FLAC by default → ./generated/run_audio_compressor/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="File or directory to compress. Defaults to your run_record_mic folder.",
    )
    parser.add_argument(
        "--codec",
        choices=["flac", "opus", "alac", "wavpack"],
        default="flac",
        help="Target codec (default: flac = lossless)",
    )
    parser.add_argument("--bitrate", type=int, default=512, help="Opus bitrate in kbps (only used with --codec opus)")
    parser.add_argument("--level", type=int, default=8, help="Compression level (FLAC: 0-12, higher = better but slower)")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    parser.add_argument("--delete", action="store_true", help="Delete original files after successful compression")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen — no changes made")

    args = parser.parse_args()
    input_path: Path = args.path.expanduser().resolve()

    if not input_path.exists():
        rprint(f"[red]Path not found:[/red] {input_path}")
        return 1

    config = build_config(args)

    # ─────────────────────────────────────────────────────────────────────
    # Dry-run mode
    # ─────────────────────────────────────────────────────────────────────
    if args.dry_run:
        mode = "Single File" if input_path.is_file() else "Directory"
        rprint(
            Panel(
                f"[bold yellow]Dry Run — {mode}[/]\n"
                f"Path: {input_path}\n"
                f"Codec: {config.codec.value.upper()}\n"
                f"Recursive: {args.recursive}\n"
                f"Output → {OUTPUT_DIR}",
                expand=False,
            )
        )
        return 0

    # ─────────────────────────────────────────────────────────────────────
    # Real compression
    # ─────────────────────────────────────────────────────────────────────
    rprint(
        Panel(
            f"[bold cyan]Compressing with FLAC (lossless)[/]\n"
            f"Source → {input_path}\n"
            f"{'Recursive scan' if args.recursive else 'Current folder only'}\n"
            f"Output → {OUTPUT_DIR}",
            expand=False,
        )
    )

    results = compress_audio(
        path=input_path,
        config=config,
        recursive=args.recursive,
        output_dir=OUTPUT_DIR,
    )

    if not results:
        rprint("[yellow]No files were compressed (none found or all skipped)[/yellow]")
        return 1

    rprint(f"\n[bold green]Success! {len(results)} file(s) compressed to FLAC[/bold green]")
    rprint(f"[green]All files saved in:[/green] {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())