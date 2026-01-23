from pathlib import Path
import shutil
from jet.audio.speech.silero.speech_analyzer import SpeechAnalyzer
from jet.file.utils import save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/results/full_recording.wav"
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay_backup/last_5_mins_1.wav"
    threshold = 0.3
    analyzer = SpeechAnalyzer(
        threshold=threshold,
        raw_threshold=0.10,
        min_duration_ms=200,
        min_std_prob=0.0,
        min_pct_threshold=10.0,
    )

    probs, energies, segments, raw_segments, num_frames = analyzer.analyze(audio_file)
    total_sec = num_frames * analyzer.step_sec

    metrics = analyzer.get_metrics(probs, segments, raw_segments, num_frames, total_sec)
    analyzer.plot_insights(probs, segments, raw_segments, num_frames, audio_file, OUTPUT_DIR)

    extra_info = {
        "total_frames": num_frames,
        "total_duration_sec": round(total_sec, 3),
        "frame_duration_ms": analyzer.frame_duration_ms,
    }

    analyzer.save_json(segments, OUTPUT_DIR, audio_file, extra_info=extra_info)
    analyzer.save_raw_json(raw_segments, OUTPUT_DIR, audio_file, extra_info=extra_info)
    analyzer.save_energies_json(energies, OUTPUT_DIR, audio_file)
    analyzer.save_segments_individually(
        audio_file,
        segments,
        OUTPUT_DIR / "segments",
        probs,
    )
    analyzer.save_raw_segments_individually(
        audio_file,
        raw_segments,
        OUTPUT_DIR,
        probs,
    )

    from rich.table import Table
    from rich.console import Console
    console = Console()
    table = Table(title=f"[bold]VAD Metrics – {Path(audio_file).name}[/bold]")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    metrics_path = OUTPUT_DIR / f"vad_metrics_{Path(audio_file).stem}.json"
    metrics_path.write_text(__import__("json").dumps(metrics, indent=2))
    print(f"Metrics JSON → {metrics_path}")

    sweep = analyzer.run_threshold_sweep(audio_file)
    analyzer.save_threshold_sweep(sweep, OUTPUT_DIR, audio_file)
    analyzer.print_threshold_sweep_table(sweep)

    print("\nAll done! Open the PNGs + overlay + std histogram + sweep table.")
    print(f"→ {OUTPUT_DIR}")

    formatted_segments = [seg.to_dict() for seg in segments]
    formatted_raw_segments = [seg.to_dict() for seg in raw_segments]

    save_file(probs, f"{str(OUTPUT_DIR)}/probs.json")
    save_file(energies, f"{str(OUTPUT_DIR)}/energies.json")
    save_file(formatted_segments, f"{str(OUTPUT_DIR)}/segments.json")
    save_file(formatted_raw_segments, f"{str(OUTPUT_DIR)}/raw_segments.json")
    save_file(metrics, f"{str(OUTPUT_DIR)}/vad_metrics.json")