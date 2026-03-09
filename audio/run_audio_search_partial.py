import shutil
from pathlib import Path

from jet.audio.audio_search import find_partial_audio_matches
from jet.audio.utils.loader import load_audio
from jet.file.utils import save_file
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

long_audio = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_per_speech/last_5_mins.wav"
short_audio = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_per_speech/segments/20260309-171141_220a46_1/sound.wav"
min_fraction = 0.5
threshold = 0.75
quick = False

console.rule("Loading audio files")
long_signal, sr_long = load_audio(long_audio)
short_signal, sr_short = load_audio(short_audio)

if sr_long != sr_short:
    console.print(
        f"[red]Warning:[/] Sample rates differ "
        f"({sr_long} Hz vs {sr_short} Hz). Using {sr_long} Hz for timing."
    )

matches = find_partial_audio_matches(
    long_signal=long_signal,
    short_signal=short_signal,
    sample_rate=sr_long,
    verbose=True,
    confidence_threshold=threshold,
    min_match_fraction=min_fraction,
    length_step_fraction=0.18 if quick else 0.10,
    max_subclips=35 if quick else 80,
    # You can expose more params later if needed (step, max_fraction, etc.)
)

table = Table(show_header=True, header_style="bold magenta")
table.add_column("#", justify="right", style="dim")
table.add_column("Start", justify="right")
table.add_column("End", justify="right")
table.add_column("Duration", justify="right")
table.add_column("Matched fraction", justify="right")
table.add_column("Confidence", justify="right", style="green")

for i, m in enumerate(matches, 1):
    start_t = m.start_sample / sr_long
    end_t = m.end_sample / sr_long
    dur = end_t - start_t
    matched_frac = m.match_length_samples / len(short_signal)

    table.add_row(
        str(i),
        f"{start_t:.3f} s",
        f"{end_t:.3f} s",
        f"{dur:.3f} s",
        f"{matched_frac:.2%}",
        f"{m.confidence:.4f}",
    )

title = "Partial matches found" if len(matches) > 1 else "Partial match found"
console.print(
    Panel(
        table,
        title=f"{title} (confidence ≥ {threshold:.2f}, matched ≥ {min_fraction:.0%} of short clip)",
        border_style="green",
        padding=(1, 2),
    )
)

save_file(matches, OUTPUT_DIR / "matches.json")
