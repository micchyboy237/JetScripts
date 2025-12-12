# JetScripts/audio/speech/run_extract_speech_chunk_scores.py
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from itertools import groupby

import torch
import matplotlib.pyplot as plt
from rich.table import Table
from rich.console import Console

from jet.audio.speech.silero.chunked_vad import SpeechProbChunk, get_speech_probabilities_chunks

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
SEGMENTS_DIR = OUTPUT_DIR / "segments"

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR.mkdir(exist_ok=True)

audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_2_speakers.wav"

# ── Load Silero VAD + audio ───────────────────────────────────────────────────
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True,
    force_reload=False,
)
(get_speech_timestamps, save_audio, read_audio, _, _) = utils

wav = read_audio(audio_path, sampling_rate=16000)

# ── Extract probabilities ─────────────────────────────────────────────────────
chunks: list[SpeechProbChunk] = get_speech_probabilities_chunks(
    audio=wav,
    model=model,
    chunk_seconds=1.0,
    overlap_seconds=0.5,
    aggregation="mean",
)

# ── Console table ─────────────────────────────────────────────────────────────
table = Table(title="Speech Confidence per 250 ms (effective step)")
table.add_column("Start (s)", justify="right")
table.add_column("End (s)", justify="right")
table.add_column("Prob", justify="center")
table.add_column("#Windows", justify="right")

for c in chunks:
    color = "green" if c["speech_prob"] > 0.7 else "yellow" if c["speech_prob"] > 0.3 else "red"
    table.add_row(
        f"{c['start_sec']:.3f}",
        f"{c['end_sec']:.3f}",
        f"[{color}]{c['speech_prob']:.3f}[/{color}]",
        str(c["num_windows"]),
    )
Console().print(table)

# ── Extract best speech segments ──────────────────────────────────────────────
def extract_speech_segments(
    chunks: list[SpeechProbChunk],
    threshold: float = 0.75,
    min_duration: float = 1.0,
) -> list[dict]:
    speech_indices = [i for i, c in enumerate(chunks) if c["speech_prob"] >= threshold]

    segments = []
    for _, group in groupby(enumerate(speech_indices), lambda x: x[0] - x[1]):
        idxs = [x[1] for x in group]
        if not idxs:
            continue
        start_chunk = chunks[idxs[0]]
        end_chunk = chunks[idxs[-1]]
        duration = end_chunk["end_sec"] - start_chunk["start_sec"]
        if duration < min_duration:
            continue

        chunk_probs = [chunks[i]["speech_prob"] for i in idxs]
        segments.append({
            "segment_id": f"segment_{len(segments)+1:03d}",
            "start_sec": round(start_chunk["start_sec"], 3),
            "end_sec": round(end_chunk["end_sec"], 3),
            "duration_sec": round(duration, 3),
            "average_speech_probability": round(float(sum(chunk_probs) / len(chunk_probs)), 4),
            "peak_speech_probability": round(float(max(chunk_probs)), 4),
            "num_chunks": len(idxs),
            "confidence_tier": (
                "high" if sum(chunk_probs)/len(chunk_probs) >= 0.9 else
                "good" if sum(chunk_probs)/len(chunk_probs) >= 0.8 else "medium"
            ),
        })
    return segments

best_segments = extract_speech_segments(chunks, threshold=0.75, min_duration=1.0)

# ── Save best_segments.json (new!) ────────────────────────────────────────────
best_segments_path = OUTPUT_DIR / "best_segments.json"
with open(best_segments_path, "w", encoding="utf-8") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_audio": os.path.basename(audio_path),
        "total_segments": len(best_segments),
        "total_speech_duration_sec": round(sum(s["duration_sec"] for s in best_segments), 3),
        "segmentation_parameters": {
            "threshold": 0.75,
            "min_duration_sec": 1.0,
            "aggregation": "mean"
        },
        "segments": best_segments
    }, f, indent=2)

# ── Save raw chunks JSON + plot + summary ─────────────────────────────────────
json_path = OUTPUT_DIR / "speech_prob_chunks.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "audio_file": os.path.basename(audio_path),
        "total_duration_sec": round(chunks[-1]["end_sec"] if chunks else 0, 4),
        "chunk_seconds": 0.5,
        "overlap_seconds": 0.25,
        "aggregation": "mean",
        "chunks": chunks,
    }, f, indent=2)

# Plot
plt.figure(figsize=(14, 4))
times = [(c["start_sec"] + c["end_sec"]) / 2 for c in chunks]
probs = [c["speech_prob"] for c in chunks]
plt.plot(times, probs, color="#1f77b4", linewidth=1.2)
plt.fill_between(times, probs, alpha=0.3, color="#1f77b4")
plt.axhline(0.75, color="green", linestyle="--", alpha=0.8, label="Segment threshold = 0.75")
plt.title(f"Silero VAD — {os.path.basename(audio_path)}")
plt.xlabel("Time (seconds)")
plt.ylabel("Speech Probability")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "speech_probability_timeline.png", dpi=300)
plt.close()

# Summary text
summary_lines = [
    f"Audio: {os.path.basename(audio_path)}",
    f"Duration: {chunks[-1]['end_sec']:.2f}s" if chunks else "0s",
    f"Detected high-confidence segments: {len(best_segments)}",
    f"Total speech time: {sum(s['duration_sec'] for s in best_segments):.2f}s",
]
with open(OUTPUT_DIR / "summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

# ── Export each segment as .wav + metadata ────────────────────────────────────
sample_rate = 16000
for idx, seg in enumerate(best_segments, start=1):
    seg_dir = SEGMENTS_DIR / f"segment_{idx:03d}"
    seg_dir.mkdir(exist_ok=True)

    start_sample = int(seg["start_sec"] * sample_rate)
    end_sample = int(seg["end_sec"] * sample_rate)
    segment_audio = wav[start_sample:end_sample]

    save_audio(str(seg_dir / "segment.wav"), segment_audio, sample_rate)

    # Update metadata with folder path
    metadata = {
        **seg,
        "folder": str(seg_dir.relative_to(OUTPUT_DIR)),
        "wav_file": "segment.wav",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(seg_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved → {seg_dir.relative_to(OUTPUT_DIR)} "
          f"[{seg['start_sec']:.2f}–{seg['end_sec']:.2f}s] "
          f"avg_prob={seg['average_speech_probability']:.3f}")

# ── Final output ──────────────────────────────────────────────────────────────
print(f"\nAll done! {len(best_segments)} high-confidence speech segments extracted.")
print(f"Results saved to: {OUTPUT_DIR.resolve()}")
print("   • best_segments.json      ← summary of clean segments")
print("   • segments/segment_XXX/   ← individual .wav + metadata")