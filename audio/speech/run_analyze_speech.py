import os
from pathlib import Path
import shutil

from jet.audio.speech.silero.speech_analyzer import SileroVADAnalyzer
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    # audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/full_recording.wav"
    threshold = 0.5

    analyzer = SileroVADAnalyzer(threshold=threshold)
    probs, segments = analyzer.analyze(audio_file)
    total_sec = len(probs) * analyzer.step_sec
    metrics = analyzer.get_metrics(probs, segments, total_sec)

    analyzer.plot_insights(probs, segments, audio_file, OUTPUT_DIR)
    analyzer.save_json(segments, OUTPUT_DIR, audio_file)
    analyzer.save_segments_individually(audio_file, segments, Path(OUTPUT_DIR) / "segments")

    # NEW: Pretty table in console
    from rich.table import Table
    from rich.console import Console
    console = Console()
    table = Table(title=f"[bold]VAD Metrics – {os.path.basename(audio_file)}[/bold]")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    save_file(probs, f"{OUTPUT_DIR}/probs.json")
    save_file(segments, f"{OUTPUT_DIR}/segments.json")
    save_file(metrics, f"{OUTPUT_DIR}/vad_metrics.json")