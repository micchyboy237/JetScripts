#!/usr/bin/env python3
"""
Realistic usage example of SpeakerIdentifier:

- Enroll known speakers with clean reference audio (meaningful labels)
- Identify unknown segments (no assumption about which segment belongs to whom)
"""

from pathlib import Path
from rich.console import Console
import logging
from rich.logging import RichHandler

from jet.audio.speech.pyannote.speaker_identifier import SpeakerIdentifier

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("speaker_id_example")
console = Console()

# ── File paths ─────────────────────────────────────────────────────────────────
refs_son = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0001/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0002/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0003/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0004/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0005/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0006/sound.wav",
]
refs_mom = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0007/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0008/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0009/sound.wav",
]

test_unknown_segment_003 = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/diarize_file/segments/segment_0003/segment.wav"
test_unknown_segment_004 = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/diarize_file/segments/segment_0004/segment.wav"

def main():
    # ── 1. Initialize ──────────────────────────────────────────────────────────
    console.rule("1. No references enrolled yet")
    identifier = SpeakerIdentifier(
        model_name="pyannote/embedding",
        similarity_threshold=0.5,
        unknown_threshold=0.4,
        cluster_when_no_refs=True,
        distance_threshold=0.7,
    )
    identifier.print_references()

    console.rule("2.1 Compare same known speakers (Son)")
    diff_refs = [refs_son[0], refs_son[1]]
    for segment_path in diff_refs:
        label, confidence = identifier.identify(segment_path)
        console.print(f"\n[bold]Segment:[/bold] {Path(segment_path).name}")
        console.print(f"→ Identified as: [green bold]{label}[/green bold] "
                      f"(confidence: {confidence:.3f})")
    
    console.rule("2.2 Compare same known speakers (Mom)")
    diff_refs = [refs_mom[0], refs_mom[1]]
    for segment_path in diff_refs:
        label, confidence = identifier.identify(segment_path)
        console.print(f"\n[bold]Segment:[/bold] {Path(segment_path).name}")
        console.print(f"→ Identified as: [green bold]{label}[/green bold] "
                      f"(confidence: {confidence:.3f})")
    
    console.rule("2.3 Compare different known speakers")
    diff_refs = [refs_son[0], refs_mom[0]]
    for segment_path in diff_refs:
        label, confidence = identifier.identify(segment_path)
        console.print(f"\n[bold]Segment:[/bold] {Path(segment_path).name}")
        console.print(f"→ Identified as: [green bold]{label}[/green bold] "
                      f"(confidence: {confidence:.3f})")

    exit()
    # ── 3. Compare n number of same known speakers (Son) ───────────────────────────────────
    console.rule("3. Compare same known speakers (Son)")
    for segment_path in refs_son:
        label, confidence = identifier.identify(segment_path)
        console.print(f"\n[bold]Segment:[/bold] {Path(segment_path).name}")
        console.print(f"→ Identified as: [green bold]{label}[/green bold] "
                      f"(confidence: {confidence:.3f})")

    # ── 4. Compare n number of same known speakers (Mom) ───────────────────────────────────
    console.rule("4. Compare same known speakers (Mom)")
    for segment_path in refs_mom:
        label, confidence = identifier.identify(segment_path)
        console.print(f"\n[bold]Segment:[/bold] {Path(segment_path).name}")
        console.print(f"→ Identified as: [green bold]{label}[/green bold] "
                      f"(confidence: {confidence:.3f})")

    # ── 5. Realistic closed-set: enroll different known speakers ───────────────
    console.rule("5. Enroll known speakers (clean references)")

    enrolled = identifier.add_reference(
        speaker_label="Son",          # ← you know this is Alice
        audio_paths=refs_son,
        force_overwrite=True
    )
    if enrolled:
        console.print("[green]Alice reference enrolled successfully[/green]")

    enrolled = identifier.add_reference(
        speaker_label="Mom",            # ← you know this is Bob
        audio_paths=refs_mom,
        force_overwrite=True
    )
    if enrolled:
        console.print("[green]Bob reference enrolled successfully[/green]")

    identifier.print_references()

    # ── 6. Identify unknown diarization segments ───────────────────────────────
    console.rule("6. Identify unknown diarization segments")

    for segment_path in [test_unknown_segment_003, test_unknown_segment_004]:
        label, confidence = identifier.identify(segment_path)
        console.print(f"\n[bold]Segment:[/bold] {Path(segment_path).name}")
        console.print(f"→ Identified as: [green bold]{label}[/green bold] "
                      f"(confidence: {confidence:.3f})")

    # ── 7. Pure open-set / clustering mode (no human labels) ──────────────────
    console.rule("7. No references → pure clustering mode")

    clustering_id = SpeakerIdentifier(
        model_name="pyannote/embedding",
        cluster_when_no_refs=True,
        distance_threshold=0.35,
    )

    for segment_path in [test_unknown_segment_003, test_unknown_segment_004]:
        label, confidence = clustering_id.identify(segment_path)
        console.print(f"\n[bold]Segment:[/bold] {Path(segment_path).name}")
        console.print(f"→ Clustered as: [blue bold]{label}[/blue bold] "
                      f"(cluster proportion: {confidence:.3f})")


if __name__ == "__main__":
    main()