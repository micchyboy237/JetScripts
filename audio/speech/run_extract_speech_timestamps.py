# JetScripts/audio/speech/run_extract_speech_timestamps.py
from collections import defaultdict
import json
from typing import Dict, Any, List
from pathlib import Path
from jet.audio.speech.silero.speech_timestamps_extractor import extract_speech_timestamps
from jet.file.utils import save_file
from rich.console import Console
import torchaudio
from silero_vad.utils_vad import read_audio
import os
import shutil

console = Console()

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def create_sub_dir(file: str):
    pass

def main(audio_file: str | Path, output_dir: str | Path, *, threshold: float = 0.5):
    audio_file = str(audio_file)
    output_dir = Path(output_dir)
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")
    segments = extract_speech_timestamps(
        audio_file,
        threshold=threshold,
        return_seconds=True,
        time_resolution=3,
    )
    waveform = read_audio(audio_file, sampling_rate=16000).unsqueeze(0)
    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments or [])}\n")
    if not segments:
        console.print("[bold yellow]No speech segments detected – skipping save.[/bold yellow]")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{seg['duration']}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan]"
        )
        folder_name = f"segment_{seg['idx']+1:03d}"  # increment idx by 1 for output subdir
        seg_dir = output_dir / folder_name
        seg_dir.mkdir(parents=True, exist_ok=True)
        save_file(seg, seg_dir / "segment.json", verbose=False)
        start_sample = int(seg['start'] * 16000)
        end_sample = int(seg['end'] * 16000)
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

        gaps: List[Dict[str, Any]] = []

        # Inter-segment silences: between consecutive speech segments
        for i in range(1, len(segments)):
            prev_end = segments[i-1]["end"]
            curr_start = segments[i]["start"]
            gap_duration_val = curr_start - prev_end
            if gap_duration_val > 0:
                gaps.append({
                    "gap_idx": i,
                    "description": f"silence between speech segment {i-1} and {i}",
                    "start": round(prev_end, 3),
                    "end": round(curr_start, 3),
                    "gap_duration": round(gap_duration_val, 3)
                })

        save_file(gaps, output_dir / "segment_gaps.json")

if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    audio_paths = [
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_2_speakers_low_prob.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/full_recording.wav",
    ]
    threshold = 0.5

    summary: Dict[str, Any] = {
        "total_files_processed": len(audio_paths),
        "files_with_speech": 0,
        "total_segments": 0,
        "per_file": defaultdict(dict),
    }
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for audio_path in audio_paths:
        output_dir = OUTPUT_DIR
        main(audio_path, output_dir, threshold=threshold)
        if (output_dir / "speech_timestamps.json").exists():
            with open(output_dir / "speech_timestamps.json") as f:
                segs = json.load(f)
            count = len(segs)
            summary["files_with_speech"] += 1
            summary["total_segments"] += count
            summary["per_file"][str(audio_path)] = {"segments": count}
    save_file(summary, Path(OUTPUT_DIR) / "summary.json")
    console.print(f"\n[bold green]Global summary saved to:[/bold green] {Path(OUTPUT_DIR)/'summary.json'}")