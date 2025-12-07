# JetScripts/audio/transcribers/run_transcribe_audio_silero.py
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from jet.audio.transcribers.utils import segments_to_srt
from jet.audio.utils import SegmentGroup, resolve_audio_paths_by_groups
from jet.file.utils import save_file
from jet.logger import logger
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry, WhisperModelsType

# ──────────────────────────────────────────────────────────────
# NEW IMPORTS for analysis/visualization
# ──────────────────────────────────────────────────────────────
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

# Output directory next to this script
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


# Set nice plotting style once
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12

TaskType = Literal["transcribe", "translate"]

def extract_all_words(segments: List[Any]) -> List[Dict[str, Any]]:
    """Flatten every word entry from all segments."""
    words = []
    for seg in segments:
        # Some APIs may use .words, others ["words"] – handle both
        seg_words = getattr(seg, "words", None) or getattr(seg, "get", lambda x: None)("words")
        if seg_words is None:  # fallback for dict-like (if needed)
            try:
                seg_words = seg["words"]
            except Exception:
                seg_words = []
        for w in seg_words:
            word_entry = {
                "segment_id": seg.id if hasattr(seg, "id") else seg.get("id"),
                "start": w.start if hasattr(w, "start") else w.get("start"),
                "end": w.end if hasattr(w, "end") else w.get("end"),
                "word": w.word.strip() if hasattr(w, "word") else w.get("word", "").strip(),
                "probability": float(w.probability if hasattr(w, "probability") else w.get("probability")),
            }
            words.append(word_entry)
    return words

class TranscriptionInsights:
    """
    Analyze Whisper transcription at word, segment (≈sentence), and phrase level.
    Classifies each segment into one of 3 confidence tiers: high / medium / low.
    """

    def __init__(self, segments: List[Any], info: Any):
        self.segments = segments
        self.info = info
        self.words = extract_all_words(segments)

    def _segment_stats(self, seg) -> Dict[str, Any]:
        word_probs = [w.probability for w in seg.words] if hasattr(seg, "words") and seg.words else []
        avg_word_prob = np.mean(word_probs) if word_probs else 0.0
        min_word_prob = np.min(word_probs) if word_probs else 0.0

        return {
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "duration": seg.end - seg.start,
            "avg_logprob": float(seg.avg_logprob),
            "no_speech_prob": float(seg.no_speech_prob),
            "compression_ratio": float(getattr(seg, "compression_ratio", 1.0)),
            "word_count": len(word_probs),
            "avg_word_prob": float(avg_word_prob),
            "min_word_prob": float(min_word_prob),
            "temperature": float(getattr(seg, "temperature", 0.0)),
        }

    def compute(self) -> Dict[str, Any]:
        seg_stats = [self._segment_stats(s) for s in self.segments]

        # === 3-tier sentence confidence classification ===
        high_conf_sentences = []
        medium_conf_sentences = []
        low_conf_sentences = []

        for s in seg_stats:
            avg_wp = s["avg_word_prob"]
            logprob = s["avg_logprob"]

            if avg_wp >= 0.92 and logprob > -0.35:
                s["confidence_tier"] = "high"
                high_conf_sentences.append(s)
            elif avg_wp < 0.75 or logprob <= -0.65:
                s["confidence_tier"] = "low"
                low_conf_sentences.append(s)
            else:
                s["confidence_tier"] = "medium"
                medium_conf_sentences.append(s)

        # === 3-tier phrase confidence classification (runs of ≥3 consecutive words) ===
        high_conf_phrases   = self._find_phrase_runs(min_prob=0.94, min_len=3)   # Extremely confident
        medium_conf_phrases = self._find_phrase_runs_between(0.75, 0.94, min_len=3)
        low_conf_phrases    = self._find_phrase_runs(max_prob=0.75, min_len=3)  # Note: < 0.75

        return {
            "file_duration": self.info.duration,
            "duration_after_vad": getattr(self.info, "duration_after_vad", None),
            "language": self.info.language,
            "language_probability": self.info.language_probability,
            "total_segments": len(seg_stats),
            "total_words": len(self.words),
            "mean_word_probability": float(np.mean([w["probability"] for w in self.words])),
            "std_word_probability": float(np.std([w["probability"] for w in self.words])),
            "median_word_probability": float(np.median([w["probability"] for w in self.words])),

            # Sentence tiers
            "high_confidence_sentences": high_conf_sentences,
            "medium_confidence_sentences": medium_conf_sentences,
            "low_confidence_sentences": low_conf_sentences,

            "count_high_confidence": len(high_conf_sentences),
            "count_medium_confidence": len(medium_conf_sentences),
            "count_low_confidence": len(low_conf_sentences),

            # Phrase tiers (new)
            "high_confidence_phrases": high_conf_phrases,
            "medium_confidence_phrases": medium_conf_phrases,
            "low_confidence_phrases": low_conf_phrases,

            "count_high_confidence_phrases": len(high_conf_phrases),
            "count_medium_confidence_phrases": len(medium_conf_phrases),
            "count_low_confidence_phrases": len(low_conf_phrases),

            "all_words": self.words,
        }

    def _find_phrase_runs(self, min_prob: float = None, max_prob: float = None, min_len: int = 3):
        phrases = []
        current = []
        for w in self.words:
            prob = w["probability"]
            if ((min_prob is not None and prob >= min_prob) or
                (max_prob is not None and prob <= max_prob)):
                current.append(w)
            else:
                if len(current) >= min_len:
                    phrases.append({
                        "start": current[0]["start"],
                        "end": current[-1]["end"],
                        "text": " ".join(c["word"] for c in current),
                        "avg_prob": float(np.mean([c["probability"] for c in current])),
                        "word_count": len(current),
                    })
                current = []
        if len(current) >= min_len:
            phrases.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(c["word"] for c in current),
                "avg_prob": float(np.mean([c["probability"] for c in current])),
                "word_count": len(current),
            })
        return phrases

    def _find_phrase_runs_between(self, min_prob: float, max_prob: float, min_len: int = 3):
        """Find runs of words where probability is in [min_prob, max_prob)."""
        phrases = []
        current = []
        for w in self.words:
            prob = w["probability"]
            if min_prob <= prob < max_prob:
                current.append(w)
            else:
                if len(current) >= min_len:
                    phrases.append({
                        "start": current[0]["start"],
                        "end": current[-1]["end"],
                        "text": " ".join(c["word"] for c in current),
                        "avg_prob": float(np.mean([c["probability"] for c in current])),
                        "word_count": len(current),
                    })
                current = []
        if len(current) >= min_len:
            phrases.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(c["word"] for c in current),
                "avg_prob": float(np.mean([c["probability"] for c in current])),
                "word_count": len(current),
            })
        return phrases

def plot_insights_enhanced(insights: Dict[str, Any], charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Professional color palette
    COLORS = {
        "high": "#16a34a",    # green-600
        "medium": "#ca8a04",  # yellow-600
        "low": "#dc2626",     # red-600
        "strong_phrase": "#22c55e",  # green-500
        "weak_phrase": "#ef4444",    # red-500
        "bg": "#1e293b",      # slate-800 (dark mode friendly)
        "text": "#e2e8f0",    # slate-200
    }

    plt.style.use("dark_background")

    # 1. Word confidence histogram
    plt.figure(figsize=(14, 7))
    probs = [w["probability"] for w in insights["all_words"]]
    sns.histplot(probs, bins=60, kde=True, color="#64748b", alpha=0.8, linewidth=0)
    plt.axvline(0.92, color=COLORS["high"], linestyle="--", linewidth=2.5, label="High threshold (≥0.92)")
    plt.axvline(0.75, color=COLORS["low"], linestyle="--", linewidth=2.5, label="Low threshold (≤0.75)")
    plt.title("Word-Level Confidence Distribution", fontsize=18, pad=20, color="white")
    plt.xlabel("Word Probability", fontsize=14, color="white")
    plt.ylabel("Count", fontsize=14, color="white")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(charts_dir / "word_confidence_histogram.png", bbox_inches="tight")
    plt.close()

    # 2. 3-Tier Confidence Timeline
    plt.figure(figsize=(18, 8))
    duration = insights["file_duration"]

    for s in insights["high_confidence_sentences"]:
        plt.axvspan(s["start"], s["end"], color=COLORS["high"], alpha=0.7,
                    label="High Confidence" if s is insights["high_confidence_sentences"][0] else "")
    for s in insights["medium_confidence_sentences"]:
        plt.axvspan(s["start"], s["end"], color=COLORS["medium"], alpha=0.5,
                    label="Medium Confidence" if s is insights["medium_confidence_sentences"][0] else "")
    for s in insights["low_confidence_sentences"]:
        plt.axvspan(s["start"], s["end"], color=COLORS["low"], alpha=0.8,
                    label="Low Confidence" if s is insights["low_confidence_sentences"][0] else "")

    plt.xlim(0, duration)
    plt.title("Sentence Confidence Timeline", fontsize=20, pad=30, color="white")
    plt.xlabel("Time (seconds)", fontsize=14, color="white")
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(charts_dir / "confidence_timeline_3tier.png", bbox_inches="tight")
    plt.close()

    # 3. Confidence tier pie chart
    plt.figure(figsize=(8, 8))
    sizes = [
        len(insights["high_confidence_sentences"]),
        len(insights["medium_confidence_sentences"]),
        len(insights["low_confidence_sentences"]),
    ]
    labels = ["High", "Medium", "Low"]
    colors = [COLORS["high"], COLORS["medium"], COLORS["low"]]
    explode = (0.05, 0.03, 0.08)

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
            textprops={'color': 'white', 'fontsize': 14}, explode=explode)
    plt.title("Sentence Confidence Distribution", fontsize=18, color="white", pad=20)
    plt.savefig(charts_dir / "confidence_pie_chart.png", bbox_inches="tight")
    plt.close()

def transcribe_segment_groups(
    audio_dir: str | Path,
    *,
    model_name: Union[str, WhisperModelsType] = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "ja",
    task: TaskType = "translate",           # ← Fixed: must be valid
    output_dir: Path | str = OUTPUT_DIR,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    # ← NEW: Control strong/weak chunk processing
    process_strong_weak_chunks: bool = False,
) -> list[Path]:
    """
    Transcribe all audio files in grouped segments (root + strong/weak chunks).
    One clean output folder per segment.
    """
    audio_dir = Path(audio_dir).resolve()
    output_base = Path(output_dir)
    shutil.rmtree(output_base, ignore_errors=True)
    output_base.mkdir(parents=True, exist_ok=True)

    console.log(f"[bold green]Loading Whisper model:[/] {model_name} → {device} ({compute_type})")
    registry = WhisperModelRegistry()
    model = registry.load_model(model_name, device=device, compute_type=compute_type)
    console.log("[bold green]Model loaded successfully[/]")

    grouped: dict[str, SegmentGroup] = resolve_audio_paths_by_groups(audio_dir)

    # --- Accurate total file count (progress bar) ---
    if process_strong_weak_chunks:
        total_files = sum(
            (1 if g["root"] else 0) + len(g["strong_chunks"]) + len(g["weak_chunks"])
            for g in grouped.values()
        )
    else:
        total_files = sum(1 if g["root"] else 0 for g in grouped.values())

    created_dirs: list[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Transcribing {task.completed}/{task.total} files..."),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Transcribing", total=total_files)

        for segment_name, group in grouped.items():
            segment_output = output_base / segment_name
            segment_output.mkdir(parents=True, exist_ok=True)
            created_dirs.append(segment_output)

            # Show segment summary
            table = Table(title=f"Segment: [bold magenta]{segment_name}[/]", show_header=True)
            table.add_column("Type", style="dim")
            table.add_column("File", style="green")
            if group["root"]:
                table.add_row("root", Path(group["root"]).name)
            if process_strong_weak_chunks:
                for f in group["strong_chunks"]:
                    table.add_row("strong", Path(f).name, style="green")
                for f in group["weak_chunks"]:
                    table.add_row("weak", Path(f).name, style="red")
            else:
                if group["strong_chunks"] or group["weak_chunks"]:
                    table.add_row(
                        "chunks",
                        f"{len(group['strong_chunks'])} strong + {len(group['weak_chunks'])} weak [dim](skipped)[/]"
                    )
            console.print(table)

            # --- Collect all audio files to process ---
            files_to_process: list[str] = []
            if group["root"]:
                files_to_process.append(group["root"])
            # Only process strong/weak chunks if explicitly enabled
            if process_strong_weak_chunks:
                files_to_process.extend(group["strong_chunks"])
                files_to_process.extend(group["weak_chunks"])
            else:
                console.log("[dim]Skipping strong/weak chunks (process_strong_weak_chunks=False)[/]")

            for audio_path_str in files_to_process:
                audio_path = Path(audio_path_str)
                rel_path = audio_path.relative_to(audio_dir)

                console.log(f"[cyan]→ Transcribing:[/] {rel_path}")

                try:
                    segments, info = model.transcribe(
                        audio=audio_path_str,
                        language=language,
                        task=task,                    # ← guaranteed valid
                        vad_filter=vad_filter,
                        word_timestamps=word_timestamps,
                        chunk_length=30,
                        beam_size=5,
                        best_of=5,
                        patience=1.0,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False,
                        initial_prompt=None,
                    )

                    all_segments = list(segments)
                    full_text = " ".join(seg.text.strip() for seg in all_segments)

                    # Output per file
                    file_out = segment_output / audio_path.stem
                    file_out.mkdir(exist_ok=True)

                    (file_out / "translation.txt").write_text(
                        f"Source: {audio_path.name}\n"
                        f"Segment: {segment_name}\n"
                        f"Model: {model_name}\n"
                        f"Task: Japanese → English ({task})\n"
                        f"Processed: {datetime.now().isoformat()}\n"
                        f"Duration: {info.duration:.2f}s\n"
                        f"Segments: {len(all_segments)}\n"
                        f"Language: {info.language} (prob: {info.language_probability:.2f})\n"
                        f"{'='*60}\n"
                        f"FULL TEXT\n"
                        f"{'='*60}\n\n"
                        f"{full_text}\n",
                        encoding="utf-8",
                    )

                    save_file(info, file_out / "info.json")
                    save_file(all_segments, file_out / "segments.json")
                    save_file(segments_to_srt(all_segments), file_out / "subtitles.srt")
                    
                    # ==== NEW: words, insights and charts ====
                    all_words = extract_all_words(all_segments)
                    save_file(all_words, file_out / "words.json")

                    insights = TranscriptionInsights(all_segments, info)
                    insights_dict = insights.compute()
                    insights_dict["all_words"] = all_words

                    # === NEW: Structured, beautiful insights directory ===
                    insights_dir = file_out / "insights"
                    insights_dir.mkdir(exist_ok=True)

                    # 1. Save main insights
                    save_file(insights_dict, insights_dir / "insights.json")

                    # 2. Readable summary.json (always saved)
                    readable_summary = {
                        "overview": {
                            "file_duration_s": round(insights_dict["file_duration"], 2),
                            "language": insights_dict["language"],
                            "language_probability": round(insights_dict["language_probability"], 4),
                            "total_segments": insights_dict["total_segments"],
                            "total_words": insights_dict["total_words"],
                            "mean_word_confidence": round(insights_dict["mean_word_probability"], 4),
                        },
                        "sentences": {
                            "high_confidence": len(insights_dict["high_confidence_sentences"]),
                            "medium_confidence": len(insights_dict["medium_confidence_sentences"]),
                            "low_confidence": len(insights_dict["low_confidence_sentences"]),
                            "high_percentage": round(len(insights_dict["high_confidence_sentences"]) / max(insights_dict["total_segments"], 1) * 100, 1),
                            "medium_percentage": round(len(insights_dict["medium_confidence_sentences"]) / max(insights_dict["total_segments"], 1) * 100, 1),
                            "low_percentage": round(len(insights_dict["low_confidence_sentences"]) / max(insights_dict["total_segments"], 1) * 100, 1),
                        },
                        "phrases": {
                            "high_confidence": len(insights_dict["high_confidence_phrases"]),
                            "medium_confidence": len(insights_dict["medium_confidence_phrases"]),
                            "low_confidence": len(insights_dict["low_confidence_phrases"]),
                        },
                        "recommendation": (
                            "Excellent quality" if len(insights_dict["high_confidence_sentences"]) == insights_dict["total_segments"]
                            else "Mostly reliable" if len(insights_dict["low_confidence_sentences"]) == 0
                            else "Needs review" if len(insights_dict["low_confidence_sentences"]) / max(insights_dict["total_segments"], 1) > 0.5
                            else "Partially reliable"
                        )
                    }
                    save_file(readable_summary, insights_dir / "summary.json")

                    # 3. Split sentences into tier-specific files (only if not empty)
                    sentences_dir = insights_dir / "sentences"
                    sentences_dir.mkdir(exist_ok=True)
                    if insights_dict["high_confidence_sentences"]:
                        save_file(insights_dict["high_confidence_sentences"],   sentences_dir / "high_confidence.json")
                    if insights_dict["medium_confidence_sentences"]:
                        save_file(insights_dict["medium_confidence_sentences"], sentences_dir / "medium_confidence.json")
                    if insights_dict["low_confidence_sentences"]:
                        save_file(insights_dict["low_confidence_sentences"],    sentences_dir / "low_confidence.json")

                    # 4. Split phrases → now 3 tiers (only if not empty)
                    phrases_dir = insights_dir / "phrases"
                    phrases_dir.mkdir(exist_ok=True)
                    if insights_dict["high_confidence_phrases"]:
                        save_file(insights_dict["high_confidence_phrases"],   phrases_dir / "high_confidence.json")
                    if insights_dict["medium_confidence_phrases"]:
                        save_file(insights_dict["medium_confidence_phrases"], phrases_dir / "medium_confidence.json")
                    if insights_dict["low_confidence_phrases"]:
                        save_file(insights_dict["low_confidence_phrases"],    phrases_dir / "low_confidence.json")

                    # 5. Charts — dedicated folder with consistent, beautiful colors
                    charts_dir = insights_dir / "charts"
                    plot_insights_enhanced(insights_dict, charts_dir)

                    # 6. Human-readable summary.txt
                    summary_path = insights_dir / "summary.txt"
                    summary_lines = [
                        "Transcription Insights Summary",
                        f"{'='*50}",
                        f"File duration        : {insights_dict['file_duration']:.2f}s",
                        f"Language (detected)  : {insights_dict['language']} (prob: {insights_dict['language_probability']:.3f})",
                        f"Total segments       : {insights_dict['total_segments']}",
                        f"Total words          : {insights_dict['total_words']}",
                        f"Mean word confidence : {insights_dict['mean_word_probability']:.3f}",
                        "",
                        "Confidence Tiers (Sentences)",
                        f"{'-'*40}",
                        f"High Confidence   (≥0.92) : {len(insights_dict['high_confidence_sentences']):3d}  ({len(insights_dict['high_confidence_sentences'])/insights_dict['total_segments']*100:5.1f}%)  [bold green]Excellent[/]",
                        f"Medium Confidence         : {len(insights_dict['medium_confidence_sentences']):3d}  ({len(insights_dict['medium_confidence_sentences'])/insights_dict['total_segments']*100:5.1f}%)  [bold yellow]Normal[/]",
                        f"Low Confidence    (≤0.75) : {len(insights_dict['low_confidence_sentences']):3d}  ({len(insights_dict['low_confidence_sentences'])/insights_dict['total_segments']*100:5.1f}%)  [bold red]Needs review[/]",
                        "",
                        "Phrases",
                        f"{'-'*40}",
                        f"High-confidence phrases (≥0.94) : {len(insights_dict['high_confidence_phrases'])}",
                        f"Medium-confidence phrases      : {len(insights_dict['medium_confidence_phrases'])}",
                        f"Low-confidence phrases  (<0.75): {len(insights_dict['low_confidence_phrases'])}",
                    ]
                    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

                    console.log(f"[bold magenta]Insights saved[/] → {file_out.name}/insights/")

                    console.log(f"[green]Saved[/] → {file_out.relative_to(output_base)}")
                    progress.advance(task_id)

                except Exception as e:
                    console.log(f"[red]Failed[/] {rel_path}: {e}")
                    logger.error(f"Transcription failed for {audio_path}: {e}")

    console.log(f"[bold green]All done![/] Outputs saved to:\n  → {output_base.resolve()}")
    return created_dirs


if __name__ == "__main__":
    AUDIO_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/silero/generated/silero_vad_stream"

    transcribe_segment_groups(
        audio_dir=AUDIO_DIR,
        model_name="large-v3",
        device="cpu",           # or "cuda" if available
        compute_type="int8",
        task="translate",       # ← explicitly set
        vad_filter=True,
        process_strong_weak_chunks=False    # False: Only process each segment root file
    )