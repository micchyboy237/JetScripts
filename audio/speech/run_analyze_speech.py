import os
from pathlib import Path
import shutil
from jet.audio.speech.silero.speech_analyzer import SileroVADAnalyzer
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/full_recording.wav"
    threshold = 0.5
    analyzer = SileroVADAnalyzer(threshold=threshold)

    probs, segments, raw_segments = analyzer.analyze(audio_file)
    total_sec = len(probs) * analyzer.step_sec

    metrics = analyzer.get_metrics(probs, segments, raw_segments, total_sec)
    analyzer.plot_insights(probs, segments, raw_segments, audio_file, OUTPUT_DIR)
    analyzer.save_json(segments, OUTPUT_DIR, audio_file)
    analyzer.save_raw_json(raw_segments, OUTPUT_DIR, audio_file)
    analyzer.save_segments_individually(audio_file, segments, Path(OUTPUT_DIR) / "segments")
    # Example: save only raw segments longer than 1.0s with high variability and mostly above threshold
    analyzer.save_raw_segments_individually(
        audio_file,
        raw_segments,
        Path(OUTPUT_DIR),
        min_duration=0.200,
        min_std_prob=0.0,
        # min_pct_threshold=10.0,
    )

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
    save_file(raw_segments, f"{OUTPUT_DIR}/raw_segments.json")
    save_file(metrics, f"{OUTPUT_DIR}/vad_metrics.json")

    # New: Threshold sweep summary
    sweep_results = analyzer.run_threshold_sweep(audio_file, thresholds=[0.3, 0.5, 0.7])
    analyzer.save_threshold_sweep(sweep_results, OUTPUT_DIR, audio_file)
    analyzer.print_threshold_sweep_table(sweep_results)