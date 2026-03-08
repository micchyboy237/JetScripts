# Shell script to create the audio_waveform package structure with full code in each file.
# Run this script from your desired parent directory.
# Usage: bash create_audio_waveform_pkg.sh

#!/bin/bash
set -e

PKG_DIR="audio_waveform"

# Create directories
mkdir -p "$PKG_DIR/audio"
mkdir -p "$PKG_DIR/vad"
mkdir -p "$PKG_DIR/ui"

# --- audio_waveform/__init__.py ---
cat > "$PKG_DIR/__init__.py" <<'EOF'
from .app import AudioWaveformWithSpeechProbApp
from .main import main

__all__ = [
    "AudioWaveformWithSpeechProbApp",
    "main",
]
EOF

# --- audio_waveform/main.py ---
cat > "$PKG_DIR/main.py" <<'EOF'
#!/usr/bin/env python3
"""
Realtime audio waveform + speech probability visualizer (entry point)
"""

import sys
from .app import AudioWaveformWithSpeechProbApp


def main():
    app = AudioWaveformWithSpeechProbApp(
        samplerate=16000,
        block_size=512,
        display_points=200,
    )
    app.start()


if __name__ == "__main__":
    sys.exit(main() or 0)
EOF

# --- audio_waveform/app.py ---
cat > "$PKG_DIR/app.py" <<'EOF'
"""
Main application logic — coordinates audio, VADs and UI updates
"""

from __future__ import annotations

import signal
import sys
import threading
from queue import Empty, Queue

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtWidgets

from .audio.circular_buffer import CircularBuffer
from .vad.silero import SileroVAD
from .vad.speechbrain import SpeechBrainVADWrapper
from .vad.firered import FireRedVADWrapper
from .ui.plots import create_plots_layout


class AudioWaveformWithSpeechProbApp:
    def __init__(
        self,
        samplerate: int = 16000,
        block_size: int = 512,
        display_points: int = 200,
    ) -> None:
        self.samplerate = samplerate
        self.block_size = block_size
        self.display_points = display_points

        # Thresholds
        self.THRES_WAVE_MEDIUM = 0.01
        self.THRES_WAVE_HIGH = 0.15
        self.THRES_PROB_MEDIUM = 0.3
        self.THRES_PROB_HIGH = 0.7

        # VAD models
        self.vad = SileroVAD(samplerate=self.samplerate)
        self.vad_sb = SpeechBrainVADWrapper()
        self.vad_fr = FireRedVADWrapper()

        # Thread-safe queue
        self.audio_queue: Queue[np.ndarray] = Queue(maxsize=50)

        # Data buffers
        self.wave_buffer = CircularBuffer(display_points)
        self.prob_buffer = CircularBuffer(display_points)
        self.prob_sb_buffer = CircularBuffer(display_points)
        self.prob_fr_buffer = CircularBuffer(display_points)

        self._init_buffers_with_zeros()

        # Qt setup
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(True)

        signal.signal(signal.SIGINT, lambda sig, frame: self.app.quit())

        pg.setConfigOptions(useOpenGL=True)

        # Create UI
        self.win, (
            self.wave_curves,
            self.prob_curves,
            self.prob_sb_curves,
            self.prob_fr_curves,
        ) = create_plots_layout()

        # Position window bottom-right
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        margin = 0
        x = screen.width() - self.win.width() - margin
        y = screen.height() - self.win.height() - 70 - margin
        self.win.move(max(0, x), max(0, y))
        self.win.show()

        # Audio input
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            callback=self._audio_callback,
        )

        # Background worker
        self._running = True
        self.worker_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
        )
        self.worker_thread.start()

        # UI update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(30)

    def _init_buffers_with_zeros(self) -> None:
        for _ in range(self.display_points):
            self.wave_buffer.append(0.0)
            self.prob_buffer.append(0.0)
            self.prob_sb_buffer.append(0.0)
            self.prob_fr_buffer.append(0.0)

    def _audio_callback(self, indata, frames, time, status) -> None:
        if status:
            print(status)
        samples = indata[:, 0].astype(np.float32)
        try:
            self.audio_queue.put_nowait(samples)
        except:
            pass  # drop if full

    def _inference_worker(self) -> None:
        while self._running:
            try:
                samples = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue

            wave_value = np.max(np.abs(samples)) if samples.size > 0 else 0.0
            self.wave_buffer.append(wave_value)

            prob = self.vad.get_speech_prob(samples)
            self.prob_buffer.append(prob)

            prob_sb = self.vad_sb.get_speech_prob(samples)
            self.prob_sb_buffer.append(prob_sb)

            prob_fr = self.vad_fr.get_speech_prob(samples)
            self.prob_fr_buffer.append(prob_fr)

    def _update_plots(self) -> None:
        self._update_one_plot(
            self.wave_buffer, self.wave_curves,
            self.THRES_WAVE_MEDIUM, self.THRES_WAVE_HIGH,
            is_waveform=True
        )

        self._update_one_plot(
            self.prob_buffer, self.prob_curves,
            self.THRES_PROB_MEDIUM, self.THRES_PROB_HIGH
        )

        self._update_one_plot(
            self.prob_sb_buffer, self.prob_sb_curves,
            self.THRES_PROB_MEDIUM, self.THRES_PROB_HIGH
        )

        self._update_one_plot(
            self.prob_fr_buffer, self.prob_fr_curves,
            self.THRES_PROB_MEDIUM, self.THRES_PROB_HIGH
        )

    def _update_one_plot(
        self,
        buffer: CircularBuffer,
        curves: tuple,
        thresh_medium: float,
        thresh_high: float,
        is_waveform: bool = False,
    ) -> None:
        data = buffer.to_array()
        if len(data) == 0:
            return

        x = np.arange(len(data), dtype=np.float32)

        low_mask = data < thresh_medium
        mid_mask = (data >= thresh_medium) & (data < thresh_high)
        high_mask = data >= thresh_high

        # Connect segments smoothly
        for mask in (low_mask, mid_mask, high_mask):
            mask[:-1] |= mask[1:]
            mask[1:] |= mask[:-1]

        low = np.where(low_mask, data, np.nan)
        mid = np.where(mid_mask, data, np.nan)
        high = np.where(high_mask, data, np.nan)

        low_curve, mid_curve, high_curve = curves
        low_curve.setData(x, low)
        mid_curve.setData(x, mid)
        high_curve.setData(x, high)

    def start(self) -> None:
        with self.stream:
            self.app.exec()
        self._running = False
EOF

# --- audio_waveform/audio/__init__.py ---
cat > "$PKG_DIR/audio/__init__.py" <<'EOF'
from .circular_buffer import CircularBuffer

__all__ = ["CircularBuffer"]
EOF

# --- audio_waveform/audio/circular_buffer.py ---
cat > "$PKG_DIR/audio/circular_buffer.py" <<'EOF'
from collections import deque
from typing import Deque

import numpy as np


class CircularBuffer:
    """Fixed-length circular buffer for scalar values."""

    def __init__(self, max_len: int) -> None:
        if max_len <= 0:
            raise ValueError("max_len must be > 0")

        self.max_len = max_len
        self._buffer: Deque[float] = deque(maxlen=max_len)

    def append(self, values: np.ndarray | float) -> None:
        if isinstance(values, np.ndarray):
            for v in values:
                self._buffer.append(float(v))
        else:
            self._buffer.append(float(values))

    def to_array(self) -> np.ndarray:
        return np.array(self._buffer, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buffer)
EOF

# --- audio_waveform/vad/__init__.py ---
cat > "$PKG_DIR/vad/__init__.py" <<'EOF'
from .silero import SileroVAD
from .speechbrain import SpeechBrainVADWrapper
from .firered import FireRedVADWrapper

__all__ = [
    "SileroVAD",
    "SpeechBrainVADWrapper",
    "FireRedVADWrapper",
]
EOF

# --- audio_waveform/vad/silero.py ---
cat > "$PKG_DIR/vad/silero.py" <<'EOF'
from __future__ import annotations

import torch
import numpy as np


class SileroVAD:
    """Thin streaming wrapper around Silero VAD"""

    def __init__(self, samplerate: int = 16000, device: str | None = None) -> None:
        if samplerate not in (8000, 16000):
            raise ValueError("Silero VAD only supports 8000 Hz or 16000 Hz")

        self.samplerate = samplerate

        self.device = torch.device(
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading Silero VAD on {self.device}... ", end="", flush=True)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model.to(self.device)
        self.model.eval()

        torch.set_num_threads(1)
        print("done.")

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if chunk.ndim != 1:
            raise ValueError("Expected 1D audio chunk")

        tensor = torch.from_numpy(chunk).float().to(self.device).unsqueeze(0)
        prob = self.model(tensor, self.samplerate).item()
        return prob
EOF

# --- audio_waveform/vad/speechbrain.py ---
cat > "$PKG_DIR/vad/speechbrain.py" <<'EOF'
from __future__ import annotations

import torch
import numpy as np


class SpeechBrainVADWrapper:
    """Streaming-like wrapper for speechbrain vad-crdnn-libriparty"""

    def __init__(self, device: str | None = None) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        print(f"Loading SpeechBrain VAD on {self.device}... ", end="", flush=True)

        from speechbrain.inference.VAD import VAD

        self.vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty",
            run_opts={"device": str(self.device)},
        )
        self.vad.eval()
        print("done.")

        self.sample_rate = 16000
        self.context_samples = int(0.5 * self.sample_rate)
        self.audio_ring = torch.zeros(
            self.context_samples, dtype=torch.float32, device=self.device
        )
        self.write_pos = 0

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return 0.0

        chunk_t = torch.from_numpy(chunk).float().to(self.device)
        chunk_len = len(chunk_t)

        space = self.context_samples - self.write_pos
        if chunk_len <= space:
            self.audio_ring[self.write_pos : self.write_pos + chunk_len] = chunk_t
            self.write_pos += chunk_len
        else:
            self.audio_ring[:chunk_len] = chunk_t[-self.context_samples :]
            self.write_pos = chunk_len

        prob_tensor = self.vad.get_speech_prob_chunk(self.audio_ring.unsqueeze(0))
        return float(prob_tensor[-1, -1].item())
EOF

# --- audio_waveform/vad/firered.py ---
cat > "$PKG_DIR/vad/firered.py" <<'EOF'
from __future__ import annotations

from pathlib import Path

import numpy as np


class FireRedVADWrapper:
    """Streaming FireRedVAD wrapper"""

    def __init__(self, device: str | None = None) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"Loading FireRedVAD **streaming** on {device}... ", end="", flush=True)

        from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

        model_dir = str(
            Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD")
            .expanduser()
            .resolve()
        )

        config = FireRedStreamVadConfig(
            use_gpu=(device == "cuda"),
            speech_threshold=0.65,
            smooth_window_size=5,
            pad_start_frame=4,
            min_speech_frame=6,
            max_speech_frame=2000,
            min_silence_frame=10,
            chunk_max_frame=30000,
        )

        self.vad = FireRedStreamVad.from_pretrained(model_dir, config=config)
        print("done.")

        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_prob = 0.0

    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return self.last_prob

        # Simple dynamic range compression / normalization
        chunk_max = np.max(np.abs(chunk)) + 1e-10
        target_peak = 0.30
        if chunk_max < 0.20:
            gain = min(target_peak / chunk_max, 8.0)
            chunk = chunk * gain
        elif chunk_max > 0.60:
            gain = 0.60 / chunk_max
            chunk = chunk * gain

        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

        if len(self.audio_buffer) < 4800:
            return self.last_prob

        to_process = self.audio_buffer[-9600:]
        results = self.vad.detect_chunk(to_process)

        self.audio_buffer = self.audio_buffer[-512:]

        if not results:
            return self.last_prob

        last = results[-1]
        prob = last.smoothed_prob
        self.last_prob = prob
        return prob
EOF

# --- audio_waveform/ui/__init__.py ---
cat > "$PKG_DIR/ui/__init__.py" <<'EOF'
from .plots import create_plots_layout

__all__ = ["create_plots_layout"]
EOF

# --- audio_waveform/ui/plots.py ---
cat > "$PKG_DIR/ui/plots.py" <<'EOF'
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


def create_plots_layout():
    """Create the four-plot layout and return window + curve tuples"""

    flags = QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.WindowStaysOnTopHint

    win = pg.GraphicsLayoutWidget(
        size=(450, 400),
        title="Realtime Audio + Speech Probability",
    )
    win.setWindowFlags(flags)

    # ── Waveform ────────────────────────────────────────────────
    wave_plot = win.addPlot()
    wave_plot.setYRange(0, 1.1)
    wave_plot.setLabel("left", "Audio Amp")
    wave_plot.showGrid(x=True, y=True, alpha=0.15)

    wave_low = wave_plot.plot(pen=pg.mkPen(150, 150, 150, width=1.2), connect="finite")
    wave_mid = wave_plot.plot(pen=pg.mkPen(0, 255, 255, width=1.8), connect="finite")
    wave_high = wave_plot.plot(pen=pg.mkPen(100, 255, 120, width=2.2), connect="finite")

    win.nextRow()

    # ── Silero Prob ─────────────────────────────────────────────
    prob_plot = win.addPlot()
    prob_plot.setYRange(0, 1)
    prob_plot.setLabel("left", "Speech Prob")
    prob_plot.showGrid(x=True, y=True, alpha=0.15)

    p_low = prob_plot.plot(pen=pg.mkPen(150, 150, 150, width=1.2), connect="finite")
    p_mid = prob_plot.plot(pen=pg.mkPen(0, 255, 255, width=1.8), connect="finite")
    p_high = prob_plot.plot(pen=pg.mkPen(100, 255, 120, width=2.2), connect="finite")

    win.nextRow()

    # ── SpeechBrain Prob ────────────────────────────────────────
    sb_plot = win.addPlot()
    sb_plot.setYRange(0, 1)
    sb_plot.setLabel("left", "SB Speech Prob")
    sb_plot.showGrid(x=True, y=True, alpha=0.15)

    sb_low = sb_plot.plot(pen=pg.mkPen(180, 150, 180, width=1.2))
    sb_mid = sb_plot.plot(pen=pg.mkPen(200, 100, 200, width=1.8))
    sb_high = sb_plot.plot(pen=pg.mkPen(220, 60, 220, width=2.2))

    win.nextRow()

    # ── FireRed Prob ────────────────────────────────────────────
    fr_plot = win.addPlot()
    fr_plot.setYRange(0, 1)
    fr_plot.setLabel("left", "FR Speech Prob")
    fr_plot.showGrid(x=True, y=True, alpha=0.15)

    fr_low = fr_plot.plot(pen=pg.mkPen(255, 200, 120, width=1.2))
    fr_mid = fr_plot.plot(pen=pg.mkPen(255, 150, 80, width=1.8))
    fr_high = fr_plot.plot(pen=pg.mkPen(255, 100, 40, width=2.2))

    wave_curves = (wave_low, wave_mid, wave_high)
    prob_curves = (p_low, p_mid, p_high)
    sb_curves = (sb_low, sb_mid, sb_high)
    fr_curves = (fr_low, fr_mid, fr_high)

    return win, (wave_curves, prob_curves, sb_curves, fr_curves)
EOF

echo "audio_waveform package structure with full code has been created in $PKG_DIR/"