# JetScripts/audio/speech/run_extract_speech_timestamps.py
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torchaudio
from jet.audio.speech.silero.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.file.utils import save_file
from jet.utils.text import format_sub_dir
from rich.console import Console
from silero_vad.utils_vad import read_audio

console = Console()

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def create_sub_dir(file: str):
    pass


def main(
    audio_file: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 0.10,
    neg_threshold: float = 0.04,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 100,
    normalize_loudness: bool = True,
):
    audio_file = str(audio_file)
    output_dir = Path(output_dir)
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")
    segments, all_speech_probs = extract_speech_timestamps(
        audio_file,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,
        time_resolution=3,
        with_scores=True,
        neg_threshold=neg_threshold,
        normalize_loudness=normalize_loudness,
    )
    waveform = read_audio(audio_file, sampling_rate=16000).unsqueeze(0)
    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments or [])}\n")
    if not segments:
        console.print(
            "[bold yellow]No speech segments detected – skipping save.[/bold yellow]"
        )
        return

    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{seg['duration']}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan]"
        )
        folder_name = (
            f"segment_{seg['num']:03d}"  # use segment number for output subdir
        )
        seg_dir = segments_dir / folder_name
        seg_dir.mkdir(parents=True, exist_ok=True)
        save_file(seg, seg_dir / "segment.json", verbose=False)
        start_sample = int(seg["start"] * 16000)
        end_sample = int(seg["end"] * 16000)
        segment_audio = waveform[:, start_sample:end_sample]
        torchaudio.save(
            str(seg_dir / "sound.wav"),
            segment_audio,
            sample_rate=16000,
            encoding="PCM_S",
            bits_per_sample=16,
        )
    save_file(segments, output_dir / "speech_timestamps.json")

    # Compute and save segment gaps

    if segments:
        gaps: list[dict[str, Any]] = []

        # Inter-segment silences: between consecutive speech segments
        for i in range(1, len(segments)):
            prev_end = segments[i - 1]["end"]
            curr_start = segments[i]["start"]
            gap_duration_val = curr_start - prev_end
            if gap_duration_val > 0:
                gaps.append(
                    {
                        "gap_idx": i,
                        "seg_idx_range": [i - 1, i],
                        "description": f"silence between speech segment {i - 1} and {i}",
                        "start": round(prev_end, 3),
                        "end": round(curr_start, 3),
                        "gap_duration": round(gap_duration_val, 3),
                    }
                )

        save_file(gaps, output_dir / "segment_gaps.json")
        save_file(all_speech_probs, output_dir / "all_speech_probs.json")


if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    audio_paths = [
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_2_speakers_short.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/preprocessors/recording_2_speakers_short_norm.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/results/full_recording.wav",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/utils/generated/run_extract_audio_segment/recording_missav.wav",
    ]

    normalize_loudness = True

    summary: dict[str, Any] = {
        "total_files_processed": len(audio_paths),
        "files_with_speech": 0,
        "total_segments": 0,
        "per_file": defaultdict(dict),
    }
    for audio_path in audio_paths:
        sub_output_dir = OUTPUT_DIR / format_sub_dir(Path(audio_path).stem)
        Path(sub_output_dir).mkdir(parents=True, exist_ok=True)
        main(audio_path, sub_output_dir, normalize_loudness=normalize_loudness)
        if (sub_output_dir / "speech_timestamps.json").exists():
            with open(sub_output_dir / "speech_timestamps.json") as f:
                segs = json.load(f)
            count = len(segs)
            summary["files_with_speech"] += 1
            summary["total_segments"] += count
            summary["per_file"][str(audio_path)] = {"segments": count}

        save_file(summary, Path(sub_output_dir) / "summary.json")
        console.print(
            f"\n[bold green]Global summary saved to:[/bold green] {Path(sub_output_dir) / 'summary.json'}"
        )
