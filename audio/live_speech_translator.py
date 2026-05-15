import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.speech.wav_utils import save_wav_file
from jet.audio.speech_detector import record_from_mic
from jet.audio.speech_handlers.base import SpeechSegmentHandler
from jet.audio.speech_handlers.speech_events import SpeechSegmentEndEvent
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_rms(audio_np: np.ndarray) -> float:
    """Root-mean-square energy of the segment audio."""
    if audio_np.size == 0:
        return 0.0
    data = audio_np.astype(np.float32)
    if np.issubdtype(audio_np.dtype, np.integer):
        data = data / np.iinfo(audio_np.dtype).max
    return float(np.sqrt(np.mean(data**2)))


def _compute_peak(audio_np: np.ndarray) -> float:
    """Peak absolute amplitude normalised to [-1, 1]."""
    if audio_np.size == 0:
        return 0.0
    data = audio_np.astype(np.float32)
    if np.issubdtype(audio_np.dtype, np.integer):
        data = data / np.iinfo(audio_np.dtype).max
    return float(np.max(np.abs(data)))


def _db(value: float) -> float:
    """Convert linear amplitude to dBFS. Returns -inf for zero."""
    return float(20 * np.log10(value)) if value > 0 else float("-inf")


def _classify_rms(rms: float) -> str:
    if rms < 0.03:
        return "very_quiet"
    if rms < 0.12:
        return "normal"
    if rms < 0.25:
        return "loud"
    return "very_loud"


# ---------------------------------------------------------------------------
# Speech handlers
# ---------------------------------------------------------------------------


def dispatch_handlers(
    handlers: list[SpeechSegmentHandler],
    speech_seg: SpeechSegment,
    seg_audio_np: np.ndarray,
    full_audio_np: np.ndarray,
    seg_dir: Path,
    seg_number: int,
    sample_rate: int,
    started_at: datetime,
) -> None:
    """Fire on_segment_end on every registered handler. Errors are caught per-handler."""
    event = SpeechSegmentEndEvent(
        segment=speech_seg,
        segment_number=seg_number,
        audio_np=seg_audio_np,
        full_audio_np=full_audio_np,
        segment_dir=seg_dir,
        started_at=started_at,
        sample_rate=sample_rate,
    )
    for handler in handlers:
        try:
            handler.on_segment_end(event)
        except Exception as exc:
            logger.error(f"Handler {type(handler).__name__} failed: {exc}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(
    speech_seg: SpeechSegment,
    seg_audio_np: np.ndarray,
    seg_number: int,
) -> dict:
    """Build a human-readable insights dict for one segment."""
    rms = _compute_rms(seg_audio_np)
    peak = _compute_peak(seg_audio_np)

    # stereo → mono shape handling
    mono = seg_audio_np
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    mono = mono.astype(np.float32)
    if np.issubdtype(seg_audio_np.dtype, np.integer):
        mono = mono / np.iinfo(seg_audio_np.dtype).max

    # zero-crossing rate
    if len(mono) > 1:
        zcr = float(np.mean(np.abs(np.diff(np.sign(mono))) > 0))
    else:
        zcr = 0.0

    # simple clipping detection
    clipping_ratio = float(np.mean(np.abs(mono) > 0.98)) if mono.size else 0.0

    summary = {
        "segment_number": seg_number,
        "timing": {
            "start_sec": round(float(speech_seg["start"]), 3),
            "end_sec": round(float(speech_seg["end"]), 3),
            "duration_sec": round(float(speech_seg["duration"]), 3),
        },
        "vad": {
            "type": speech_seg.get("type", "speech"),
            "prob": round(float(speech_seg.get("prob", 0.0)), 4),
            "end_reason": speech_seg.get("end_reason", None),
            "frames_length": speech_seg.get("frames_length", 0),
            "frame_start": speech_seg.get("frame_start", 0),
            "frame_end": speech_seg.get("frame_end", 0),
        },
        "audio": {
            "samples": int(seg_audio_np.size),
            "rms": round(rms, 6),
            "rms_db": round(_db(rms), 2),
            "peak": round(peak, 6),
            "peak_db": round(_db(peak), 2),
            "zero_crossing_rate": round(zcr, 4),
            "clipping_ratio": round(clipping_ratio, 6),
            "level_class": _classify_rms(rms),
        },
        "flags": {
            "is_clipping": clipping_ratio > 0.001,
            "is_very_quiet": rms < 0.03,
            "is_loud": rms >= 0.12,
            "high_zcr": zcr > 0.3,  # lots of sign changes → noisy/fricative
        },
    }

    # optional per-frame probs
    if speech_seg.get("segment_probs"):
        probs = speech_seg["segment_probs"]
        summary["vad"]["prob_min"] = round(float(min(probs)), 4)
        summary["vad"]["prob_max"] = round(float(max(probs)), 4)
        summary["vad"]["prob_std"] = round(float(np.std(probs)), 4)

    return summary


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_COLORS = {
    "waveform": "#4A90D9",
    "rms_line": "#E67E22",
    "prob_fill": "#2ECC71",
    "prob_line": "#27AE60",
    "grid": "#E0E0E0",
    "bg": "#FAFAFA",
    "panel_bg": "#FFFFFF",
}


def save_segment_plot(
    speech_seg: SpeechSegment,
    seg_audio_np: np.ndarray,
    seg_number: int,
    out_path: Path,
    sample_rate: int = 16000,
) -> None:
    """
    Save a 3-panel diagnostic plot for one speech segment:
      Panel 1 — Waveform with RMS envelope
      Panel 2 — Short-time RMS energy (10 ms frames)
      Panel 3 — Per-frame VAD probabilities (if available)
    """
    mono = seg_audio_np
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    mono = mono.astype(np.float32)
    if np.issubdtype(seg_audio_np.dtype, np.integer):
        mono = mono / np.iinfo(seg_audio_np.dtype).max

    n_samples = len(mono)
    time_axis = np.linspace(0, n_samples / sample_rate, n_samples)

    # short-time RMS (10 ms frames)
    frame_len = max(1, int(0.010 * sample_rate))
    n_frames = n_samples // frame_len
    if n_frames > 0:
        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms_frames = np.sqrt(np.mean(frames**2, axis=1))
        rms_time = np.arange(n_frames) * frame_len / sample_rate
    else:
        rms_frames = np.array([])
        rms_time = np.array([])

    probs = speech_seg.get("segment_probs", [])
    has_probs = bool(probs)
    n_panels = 3 if has_probs else 2

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10, 3 * n_panels),
        sharex=False,
        facecolor=_COLORS["bg"],
    )
    if n_panels == 1:
        axes = [axes]

    start_s = float(speech_seg["start"])
    end_s = float(speech_seg["end"])
    prob = float(speech_seg.get("prob", 0.0))
    duration = float(speech_seg["duration"])

    fig.suptitle(
        f"Segment {seg_number}  |  {start_s:.2f}s → {end_s:.2f}s"
        f"  |  dur {duration:.2f}s  |  VAD prob {prob:.2f}",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    # ── Panel 1: Waveform ──────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor(_COLORS["panel_bg"])
    ax0.plot(time_axis, mono, color=_COLORS["waveform"], lw=0.6, alpha=0.85)
    if len(rms_frames):
        ax0.plot(
            rms_time,
            rms_frames,
            color=_COLORS["rms_line"],
            lw=1.5,
            label="RMS envelope",
            zorder=3,
        )
        ax0.plot(rms_time, -rms_frames, color=_COLORS["rms_line"], lw=1.5, zorder=3)
    ax0.axhline(0, color="#CCCCCC", lw=0.8, zorder=0)
    ax0.set_ylabel("Amplitude", fontsize=9)
    ax0.set_ylim(-1.05, 1.05)
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, color=_COLORS["grid"], lw=0.5)
    ax0.set_title("Waveform + RMS Envelope", fontsize=9)
    ax0.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2fs"))

    # ── Panel 2: Short-time RMS energy ────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor(_COLORS["panel_bg"])
    if len(rms_frames):
        ax1.fill_between(
            rms_time,
            rms_frames,
            alpha=0.4,
            color=_COLORS["rms_line"],
            label="RMS energy",
        )
        ax1.plot(rms_time, rms_frames, color=_COLORS["rms_line"], lw=1.2)
    ax1.set_ylabel("RMS", fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, color=_COLORS["grid"], lw=0.5)
    ax1.set_title("Short-time RMS Energy (10 ms frames)", fontsize=9)
    ax1.set_xlabel("Time (s)" if not has_probs else "", fontsize=9)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2fs"))

    # ── Panel 3: VAD probabilities (optional) ─────────────────────────────
    if has_probs:
        ax2 = axes[2]
        ax2.set_facecolor(_COLORS["panel_bg"])
        prob_time = np.linspace(0, duration, len(probs))
        ax2.fill_between(prob_time, probs, alpha=0.3, color=_COLORS["prob_fill"])
        ax2.plot(prob_time, probs, color=_COLORS["prob_line"], lw=1.4, label="VAD prob")
        ax2.axhline(prob, color="#E74C3C", lw=1.0, ls="--", label=f"avg {prob:.2f}")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Probability", fontsize=9)
        ax2.set_xlabel("Time (s)", fontsize=9)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, color=_COLORS["grid"], lw=0.5)
        ax2.set_title("Per-frame VAD Probabilities", fontsize=9)
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2fs"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main save routine
# ---------------------------------------------------------------------------


def save_segment_data(
    speech_seg: SpeechSegment,
    seg_audio_np: np.ndarray,
    sample_rate: int = 16000,
) -> tuple[Path, int]:
    segment_root = OUTPUT_DIR / "segments"
    segment_root.mkdir(parents=True, exist_ok=True)

    existing = sorted(segment_root.glob("segment_*"))
    used_numbers = {
        int(seg.name.split("_")[1])
        for seg in existing
        if seg.name.split("_")[1].isdigit()
    }
    seg_number = 1
    while seg_number in used_numbers:
        seg_number += 1

    seg_dir = segment_root / f"segment_{seg_number:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. WAV ─────────────────────────────────────────────────────────────
    wav_path = seg_dir / "sound.wav"
    seg_sound_file = save_wav_file(wav_path, seg_audio_np)

    # ── 2. Metadata (raw VAD output) ───────────────────────────────────────
    metadata_path = seg_dir / "metadata.json"
    metadata_path.write_text(json.dumps(speech_seg, indent=2), encoding="utf-8")

    # ── 3. Summary (human-readable insights) ──────────────────────────────
    summary = build_summary(speech_seg, seg_audio_np, seg_number)
    summary_path = seg_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── 4. Plot ────────────────────────────────────────────────────────────
    plot_path = seg_dir / "plot.png"
    save_segment_plot(speech_seg, seg_audio_np, seg_number, plot_path, sample_rate)

    logger.success(f"Segment {seg_number} saved to: {seg_dir}")
    for p in (seg_sound_file, metadata_path, summary_path, plot_path):
        logger.success(str(p), bright=True)

    return seg_dir, seg_number


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from jet.audio.helpers.silence import SAMPLE_RATE
    from jet.audio.speech_handlers.websocket_subtitle_sender import (
        WebsocketSubtitleSender,
    )

    duration_seconds = None
    trim_silent = False
    quit_on_silence = False

    handlers: list[SpeechSegmentHandler] = [
        # add or remove handlers here
        WebsocketSubtitleSender(
            global_srt_path=OUTPUT_DIR,
        ),
    ]
    recording_started_at = datetime.now(timezone.utc)

    data_stream = record_from_mic(
        duration_seconds,
        trim_silent=trim_silent,
        quit_on_silence=quit_on_silence,
    )

    segments: list[dict] = []

    for speech_seg, seg_audio_np, full_audio_np in data_stream:
        # start / end / duration are already in seconds — no conversion needed
        seg_dir, seg_number = save_segment_data(
            speech_seg, seg_audio_np, sample_rate=SAMPLE_RATE
        )
        dispatch_handlers(
            handlers,
            speech_seg,
            seg_audio_np,
            full_audio_np,
            seg_dir,
            seg_number,
            SAMPLE_RATE,
            recording_started_at,
        )
        segments.append(speech_seg)
        save_file(segments, OUTPUT_DIR / "all_segments.json", verbose=False)
        save_wav_file(OUTPUT_DIR / "full_recording.wav", full_audio_np)

    for h in handlers:
        if hasattr(h, "close"):
            h.close()
