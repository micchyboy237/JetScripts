from jet.audio.speech.pyannote.segment_speaker_assigner import SegmentSpeakerAssigner
from jet.audio.utils import resolve_audio_paths
from jet.file.utils import save_file

from pathlib import Path
from rich.console import Console
from rich.table import Table

import shutil

BASE_OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

segments_dir = Path(
    "~/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay_2_speakers/segments"
).expanduser()

console = Console()

def main():
    all_segments = resolve_audio_paths(segments_dir, recursive=True)
    if not all_segments:
        console.print("[bold red]No audio segments found in directory![/bold red]")
        console.print(f"Looked in: {segments_dir}")
        return

    # Take first 8 segments (or all if fewer) for demo
    demo_segments = all_segments[:8]
    console.print(f"[bold cyan]Found {len(all_segments)} segments total — using first {len(demo_segments)} for demo[/bold cyan]\n")

    assigner = SegmentSpeakerAssigner()

    # Collect info for final table
    segment_log = []
    segment_results = []

    for i, segment_path in enumerate(demo_segments, 1):
        console.rule(f"Segment {i} — {Path(segment_path).name}")
        similarity_result, assignment_meta = assigner.add_segment(segment_path)
        console.print(f"[bold]Assigned speaker:[/bold] {assignment_meta['label']}")
        console.print(f"[bold]Segment number:[/bold] {assignment_meta['segment_num']}")
        console.print(f"[bold]Same as previous?:[/bold] {'[green]Yes[/green]' if assignment_meta['is_same_speaker'] else '[yellow]No[/yellow]'}")
        if assignment_meta['prev_speaker_label'] is not None:
            console.print(f"[bold]Previous speaker:[/bold] {assignment_meta['prev_speaker_label']}")
        console.print("")

        # Shorten path: last two components
        path_parts = Path(segment_path).parts
        short_path = "/".join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]

        segment_log.append({
            "num": assignment_meta["segment_num"],
            "label": assignment_meta["label"],
            "path": short_path,
        })

        segment_results.append({
            "assignment_meta": assignment_meta,
            "similarity_result": similarity_result,
        })

    # ── Final tables ─────────────────────────────────────────────────────

    # 1. Per-segment assignment log
    detail_table = Table(title="Segment Assignment Log", show_header=True, header_style="bold cyan")
    detail_table.add_column("Segment #", style="dim", justify="right")
    detail_table.add_column("Speaker", style="magenta", justify="center")
    detail_table.add_column("Path (last two)", style="green", no_wrap=False)

    for entry in segment_log:
        detail_table.add_row(
            str(entry["num"]),
            str(entry["label"]),
            entry["path"]
        )

    console.print(detail_table)
    console.print("")  # spacing

    # 2. Speaker summary (existing)
    speaker_counts = assigner.get_speaker_counts()

    table = Table(title="Final Speaker Statistics")
    table.add_column("Speaker Label", style="cyan", no_wrap=True)
    table.add_column("Segment Count", justify="right", style="magenta")

    for label, count in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True):
        table.add_row(str(label), str(count))
    console.print(table)
    console.print(f"\n[italic dim]Total unique speakers detected: {len(speaker_counts)}[/italic dim]")

    save_file(segment_results, BASE_OUTPUT_DIR / "results.json")
    save_file(speaker_counts, BASE_OUTPUT_DIR / "speaker_counts.json")
    save_file(assigner.speaker_labels, BASE_OUTPUT_DIR / "speaker_labels.json")

if __name__ == "__main__":
    main()
