Ctrl+K
Ctrl+J
Query
Evaluate speech stats code and summary results.
Improve to reuse existing code such as from "jet.audio.helpers.energy"
Saved summary results
Segment 1
summary = {
"segment_id": 1,
"start_frame": 975,
"end_frame": 1568,
"start_time_sec": 9.74,
"end_time_sec": 15.5,
"duration_sec": 5.76,
"audio_samples": 92160,
"prob_frames": 594,
"forced_split": true,
"trigger_reason": "valley_detection",
"started_at": "2026-03-29T21:41:47.355466",
"segment_rms": 0.05348586633792529,
"loudness": "normal",
"has_sound": true,
"stats": {
"avg_smoothed_prob": 0.922,
"max_smoothed_prob": 1.0,
"min_smoothed_prob": 0.403,
"prob_std": 0.15,
"total_frames": 594,
"speech_frames": 594,
"speech_frame_ratio": 1.0,
"silence_ratio": 0.0,
"internal_silence_frames": 0,
"max_consecutive_silence_frames": 0,
"has_internal_pause": false,
"max_consecutive_speech_frames": 594,
"avg_consecutive_speech_frames": 594.0,
"speech_segments_count": 1,
"long_speech_ratio": 1.0,
"avg_rms": 0.0,
"rms_std": 0.0,
"speech_rms_mean": 0.0,
"silence_rms_mean": 0.0,
"energy_speech_correlation": 0.0,
"low_energy_speech_ratio": 0.0,
"high_energy_non_speech_ratio": 0.0,
"prob_velocity_mean": 0.0063,
"prob_velocity_std": 0.0135,
"prob_acceleration_mean": 0.0041,
"speech_transition_rate": 0.0,
"estimated_speech_duration_sec": 5.94,
"confidence_level": "high",
"quality_score": 0.165
}
}
Segment 2
summary = {
"segment_id": 2,
"start_frame": 1569,
"end_frame": 2524,
"start_time_sec": 15.68,
"end_time_sec": 24.896,
"duration_sec": 9.216,
"audio_samples": 147456,
"prob_frames": 956,
"forced_split": false,
"trigger_reason": "non_speech",
"started_at": "2026-03-29T21:41:53.611056",
"segment_rms": 0.05480782839758654,
"loudness": "normal",
"has_sound": true,
"stats": {
"avg_smoothed_prob": 0.966,
"max_smoothed_prob": 1.0,
"min_smoothed_prob": 0.0,
"prob_std": 0.145,
"total_frames": 956,
"speech_frames": 936,
"speech_frame_ratio": 0.979,
"silence_ratio": 0.021,
"internal_silence_frames": 20,
"max_consecutive_silence_frames": 20,
"has_internal_pause": true,
"max_consecutive_speech_frames": 936,
"avg_consecutive_speech_frames": 936.0,
"speech_segments_count": 1,
"long_speech_ratio": 0.979,
"avg_rms": 0.0,
"rms_std": 0.0,
"speech_rms_mean": 0.0,
"silence_rms_mean": 0.0,
"energy_speech_correlation": 0.0,
"low_energy_speech_ratio": 0.0,
"high_energy_non_speech_ratio": 0.0,
"prob_velocity_mean": 0.0026,
"prob_velocity_std": 0.0124,
"prob_acceleration_mean": 0.0015,
"speech_transition_rate": 0.001,
"estimated_speech_duration_sec": 9.36,
"confidence_level": "high",
"quality_score": 0.242
}
}
Instructions
Provide step by step analysis first.
Show unified diff for updated files, while show python code block for new files.
Existing Files Contents

# jet_python_modules/jet/audio/audio_waveform/hybrid_stream_vad_postprocessor.py

from collections import deque
from fireredvad.core.stream_vad_postprocessor import (
StreamVadFrameResult,
StreamVadPostprocessor,
VadState,
)
from rich.console import Console
console = Console()
class HybridStreamVadPostprocessor(StreamVadPostprocessor):
def **init**(
self,
smooth_window_size: int = 5,
speech_threshold: float = 0.5,
pad_start_frame: int = 5,
min_speech_frame: int = 30,
soft_max_speech_frame: int = 450,
hard_max_speech_frame: int = 800,
min_silence_frame: int = 20,
search_window: int = 200,
valley_threshold: float = 0.65,
min_valley_consecutive_frames: int = 5,
):
self.soft_limit = soft_max_speech_frame
self.hard_limit = hard_max_speech_frame
self.search_window = search_window
self.valley_threshold = valley_threshold
self.min_valley_consecutive_frames = min_valley_consecutive_frames
self.\_last_state = None
self.\_last_speech_cnt = -1
self.\_last_silence_cnt = -1
self.\_last_force_split_frame = -1
self.recent_probs: deque[float] = deque(maxlen=1024)
self.last_force_split_reason = None
super().**init**(
smooth_window_size,
speech_threshold,
pad_start_frame,
min_speech_frame,
hard_max_speech_frame,
min_silence_frame,
)
@property
def was_force_splitted(self) -> bool:
return self.last_force_split_reason in ("valley_detection", "hard_limit")
@property
def last_split_reason(self) -> str | None:
return self.last_force_split_reason
def reset(self):
super().reset()
if hasattr(self, "recent_probs"):
self.recent_probs.clear()
else:
self.recent_probs = deque(maxlen=1024)
self.\_last_state = None
self.\_last_speech_cnt = -1
self.\_last_silence_cnt = -1
self.\_last_force_split_frame = -1
def process_one_frame(self, raw_prob: float) -> StreamVadFrameResult:
assert 0.0 <= raw_prob <= 1.0
self.frame_cnt += 1
smoothed_prob = self.smooth_prob(raw_prob)
self.recent_probs.append(smoothed_prob)
is_speech = bool(self.apply_threshold(smoothed_prob))
result = StreamVadFrameResult(
frame_idx=self.frame_cnt,
is_speech=is_speech,
raw_prob=round(raw_prob, 3),
smoothed_prob=round(smoothed_prob, 3),
)
return self.state_transition(is_speech, result)
def \_has_valid_valley(self, window: list[float]) -> bool:
"""
True if there exists a consecutive run of frames
below valley_threshold long enough to be meaningful.
"""
count = 0
max_count = 0
for p in window:
if p < self.valley_threshold:
count += 1
if count > max_count:
max_count = count
else:
count = 0
return max_count >= self.min_valley_consecutive_frames
def state_transition(
self, is_speech: bool, result: StreamVadFrameResult
) -> StreamVadFrameResult:
prev_state = self.state
prev_speech = self.speech_cnt
prev_silence = self.silence_cnt
if self.hit_max_speech and is_speech:
result.is_speech_start = True
console.print(
f"[RECOVERY] {self.frame_cnt:5d} | {self.state.name} → SPEECH START (hit max recovery)",
style="cyan bold",
)
result.speech_start_frame = self.frame_cnt
self.last_speech_start_frame = result.speech_start_frame
self.hit_max_speech = False
if self.state == VadState.SILENCE:
if is_speech:
self.state = VadState.POSSIBLE_SPEECH
self.speech_cnt += 1
else:
self.silence_cnt += 1
self.speech_cnt = 0
elif self.state == VadState.POSSIBLE_SPEECH:
if is_speech:
self.speech_cnt += 1
valid_start = True
if self.speech_cnt >= self.min_speech_frame:
recent = list(self.recent_probs)[-self.min_speech_frame :]
if recent:
speech_frames = sum(p >= self.speech_threshold for p in recent)
speech_ratio = speech_frames / len(recent)
if speech_ratio < 0.6:
valid_start = False
if not valid_start:
self.state = VadState.SILENCE
self.speech_cnt = 0
self.silence_cnt = 1
return result
self.state = VadState.SPEECH
result.is_speech_start = True
console.print(
f"[START] {self.frame_cnt:5d} | POSSIBLE_SPEECH → SPEECH (cnt={self.speech_cnt})",
style="green bold",
)
result.speech_start_frame = max(
1,
self.frame_cnt - self.speech_cnt + 1 - self.pad_start_frame,
self.last_speech_end_frame + 1,
)
self.last_speech_start_frame = result.speech_start_frame
self.silence_cnt = 0
else:
self.state = VadState.SILENCE
self.silence_cnt = 1
self.speech_cnt = 0
elif self.state == VadState.SPEECH:
self.speech_cnt += 1
if is_speech:
self.silence_cnt = 0
force_split = False
window = []
if self.speech_cnt >= self.hard_limit:
force_split = True
elif (
self.speech_cnt > self.soft_limit
and len(self.recent_probs) >= self.search_window
):
window = list(self.recent_probs)[-self.search_window :]
if self.\_has_valid_valley(window):
force_split = True
if force_split:
min_prob_str = (
f"min_prob={min(window):.3f}" if window else "hard limit"
)
console.print(
f"[SPLIT] {self.frame_cnt:5d} | SPEECH → END ({min_prob_str}, cnt={self.speech_cnt})",
style="bold red",
)
console.print(
f" soft={self.soft_limit} hard={self.hard_limit}",
style="dim magenta",
)
self.hit_max_speech = True
self.speech_cnt = 0
result.is_speech_end = True
self.last_force_split_reason = (
"valley_detection" if window else "hard_limit"
)
result.speech_end_frame = self.frame_cnt
result.speech_start_frame = self.last_speech_start_frame
self.last_speech_start_frame = -1
self.last_speech_end_frame = result.speech_end_frame
else:
self.state = VadState.POSSIBLE_SILENCE
self.silence_cnt += 1
elif self.state == VadState.POSSIBLE_SILENCE:
self.speech_cnt += 1
if is_speech:
self.state = VadState.SPEECH
self.silence_cnt = 0
if self.speech_cnt >= self.hard_limit:
self.hit_max_speech = True
self.speech_cnt = 0
result.is_speech_end = True
result.speech_end_frame = self.frame_cnt
result.speech_start_frame = self.last_speech_start_frame
self.last_speech_start_frame = -1
self.last_speech_end_frame = result.speech_end_frame
console.print(
f"[STATE] {self.frame_cnt:5d} | "
f"POSSIBLE_SILENCE → SPEECH (continued) → hit hard limit → END",
style="bold magenta",
)
else:
self.silence_cnt += 1
if self.speech_cnt < self.soft_limit:
return result
if self.silence_cnt >= self.min_silence_frame:
self.state = VadState.SILENCE
result.is_speech_end = True
self.last_force_split_reason = "non_speech"
result.speech_end_frame = self.frame_cnt
result.speech_start_frame = self.last_speech_start_frame
self.last_speech_end_frame = result.speech_end_frame
console.print(
f"[END] {self.frame_cnt:5d} | POSSIBLE_SILENCE → SILENCE (after {self.silence_cnt} silent, speech was {self.frame_cnt - self.last_speech_start_frame} frames)",
style="blue bold",
)
self.last_speech_start_frame = -1
self.speech_cnt = 0
self.\_log_if_changed(prev_state, prev_speech, prev_silence, result)
return result
def \_log_if_changed(self, prev_state, prev_speech, prev_silence, result):
"""Only print when state or important counters change"""
if self.state != prev_state:
if self.state == VadState.SILENCE:
console.print(
f"[silence] {self.frame_cnt:5d} entered silence (after {prev_silence} transition frames)",
style="dim bright_black",
)
style = {
VadState.SPEECH: "green bold",
VadState.POSSIBLE_SILENCE: "yellow",
VadState.POSSIBLE_SPEECH: "cyan",
VadState.SILENCE: "dim bright_black",
}.get(self.state, "white")
console.print(
f"[STATE] {self.frame_cnt:5d} | {prev_state.name:13} → {self.state.name:13}",
style=style,
)
if result.is_speech_start or result.is_speech_end:
pass
self.\_last_state = self.state
self.\_last_speech_cnt = self.speech_cnt
self.\_last_silence_cnt = self.silence_cnt

# jet_python_modules/jet/audio/audio_waveform/speech_events.py

from **future** import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from jet.audio.audio_waveform.speech_stats import SpeechSegmentStats
from jet.audio.audio_waveform.speech_types import SpeechFrame
@dataclass
class SpeechSegmentStartEvent:
segment_id: int
start_frame: int
start_time_sec: float
started_at: str
segment_dir: Path | None = None
@dataclass
class SpeechSegmentEndEvent:
segment_id: int
start_frame: int
end_frame: int
start_time_sec: float
end_time_sec: float
duration_sec: float
audio: np.ndarray
prob_frames: list[SpeechFrame]
forced_split: bool
trigger_reason: str
started_at: str
segment_rms: float
loudness: str
has_sound: bool
stats: SpeechSegmentStats | None = None
segment_dir: Path | None = None

# jet_python_modules/jet/audio/audio_waveform/speech_stats.py

from **future** import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, List
import numpy as np
if TYPE_CHECKING:
from .speech_events import SpeechSegmentEndEvent
from .speech_types import SpeechFrame
@dataclass
class SpeechSegmentStats:
avg_smoothed_prob: float
max_smoothed_prob: float
min_smoothed_prob: float
prob_std: float
total_frames: int
speech_frames: int
speech_frame_ratio: float
silence_ratio: float
internal_silence_frames: int
max_consecutive_silence_frames: int
has_internal_pause: bool
max_consecutive_speech_frames: int
avg_consecutive_speech_frames: float
speech_segments_count: int
long_speech_ratio: float
avg_rms: float
rms_std: float
speech_rms_mean: float
silence_rms_mean: float
energy_speech_correlation: float
low_energy_speech_ratio: float
high_energy_non_speech_ratio: float
prob_velocity_mean: float
prob_velocity_std: float
prob_acceleration_mean: float
speech_transition_rate: float
estimated_speech_duration_sec: float
confidence_level: str
quality_score: float
def derive_segment_stats(event: SpeechSegmentEndEvent) -> SpeechSegmentStats:
prob_frames: List[SpeechFrame] = event.prob_frames or []
if not prob_frames:
return \_empty_stats()
smoothed_probs = np.array(
[f["smoothed_prob"] for f in prob_frames], dtype=np.float32
)
is_speech = np.array([f["is_speech"] for f in prob_frames], dtype=np.int32)
rms_values = \_extract_rms(prob_frames)
avg_prob = float(np.mean(smoothed_probs))
max_prob = float(np.max(smoothed_probs))
min_prob = float(np.min(smoothed_probs))
prob_std = float(np.std(smoothed_probs)) if len(smoothed_probs) > 1 else 0.0
total_frames = len(prob_frames)
speech_frames = int(np.sum(is_speech))
speech_ratio = speech_frames / total_frames if total_frames else 0.0
silence_ratio = 1.0 - speech_ratio
max_silence_run = \_max_consecutive_run(is_speech == 0)
has_pause = max_silence_run >= 5
speech_runs = \_collect_runs(is_speech == 1)
max_speech_run = max(speech_runs) if speech_runs else 0
avg_speech_run = float(np.mean(speech_runs)) if speech_runs else 0.0
speech_segments_count = len(speech_runs)
long_speech_ratio = (
sum(r for r in speech_runs if r >= 20) / total_frames if total_frames else 0.0
)
avg_rms = float(np.mean(rms_values))
rms_std = float(np.std(rms_values)) if len(rms_values) > 1 else 0.0
speech_rms_mean = (
float(np.mean(rms_values[is_speech == 1])) if np.any(is_speech == 1) else 0.0
)
silence_rms_mean = (
float(np.mean(rms_values[is_speech == 0])) if np.any(is_speech == 0) else 0.0
)
energy_corr = \_safe_corr(smoothed_probs, rms_values)
low_energy_speech_ratio = float(
np.mean((is_speech == 1) & (rms_values < avg_rms _ 0.5))
)
high_energy_non_speech_ratio = float(
np.mean((is_speech == 0) & (rms_values > avg_rms _ 1.5))
)
velocity = np.diff(smoothed_probs)
acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([])
prob_velocity_mean = float(np.mean(np.abs(velocity))) if len(velocity) else 0.0
prob_velocity_std = float(np.std(velocity)) if len(velocity) else 0.0
prob_acceleration_mean = (
float(np.mean(np.abs(acceleration))) if len(acceleration) else 0.0
)
transitions = np.sum(np.abs(np.diff(is_speech)))
transition_rate = float(transitions / total_frames) if total_frames else 0.0
estimated_speech_sec = speech_frames _ 0.010
confidence_level = \_confidence_from_prob(avg_prob)
quality_score = \_compute_quality_score(
avg_prob=avg_prob,
speech_ratio=speech_ratio,
prob_std=prob_std,
energy_corr=energy_corr,
transition_rate=transition_rate,
forced_split=event.forced_split,
)
return SpeechSegmentStats(
avg_smoothed_prob=round(avg_prob, 3),
max_smoothed_prob=round(max_prob, 3),
min_smoothed_prob=round(min_prob, 3),
prob_std=round(prob_std, 3),
total_frames=total_frames,
speech_frames=speech_frames,
speech_frame_ratio=round(speech_ratio, 3),
silence_ratio=round(silence_ratio, 3),
internal_silence_frames=total_frames - speech_frames,
max_consecutive_silence_frames=max_silence_run,
has_internal_pause=has_pause,
max_consecutive_speech_frames=max_speech_run,
avg_consecutive_speech_frames=round(avg_speech_run, 3),
speech_segments_count=speech_segments_count,
long_speech_ratio=round(long_speech_ratio, 3),
avg_rms=round(avg_rms, 5),
rms_std=round(rms_std, 5),
speech_rms_mean=round(speech_rms_mean, 5),
silence_rms_mean=round(silence_rms_mean, 5),
energy_speech_correlation=round(energy_corr, 3),
low_energy_speech_ratio=round(low_energy_speech_ratio, 3),
high_energy_non_speech_ratio=round(high_energy_non_speech_ratio, 3),
prob_velocity_mean=round(prob_velocity_mean, 4),
prob_velocity_std=round(prob_velocity_std, 4),
prob_acceleration_mean=round(prob_acceleration_mean, 4),
speech_transition_rate=round(transition_rate, 4),
estimated_speech_duration_sec=round(estimated_speech_sec, 3),
confidence_level=confidence_level,
quality_score=quality_score,
)
def \_extract_rms(prob_frames: List[SpeechFrame]) -> np.ndarray:
"""Fallback-safe RMS extraction."""
if "rms" in prob_frames[0]:
return np.array([f["rms"] for f in prob_frames], dtype=np.float32)
return np.zeros(len(prob_frames), dtype=np.float32)
def \_max_consecutive_run(mask: np.ndarray) -> int:
max_run = 0
current = 0
for val in mask:
if val:
current += 1
max_run = max(max_run, current)
else:
current = 0
return max_run
def \_collect_runs(mask: np.ndarray) -> List[int]:
runs = []
current = 0
for val in mask:
if val:
current += 1
else:
if current > 0:
runs.append(current)
current = 0
if current > 0:
runs.append(current)
return runs
def \_safe_corr(a: np.ndarray, b: np.ndarray) -> float:
if len(a) < 2 or len(b) < 2:
return 0.0
if np.std(a) == 0 or np.std(b) == 0:
return 0.0
return float(np.corrcoef(a, b)[0, 1])
def \_confidence_from_prob(avg_prob: float) -> str:
if avg_prob >= 0.75:
return "high"
elif avg_prob >= 0.5:
return "medium"
return "low"
def \_compute_quality_score(
_,
avg_prob: float,
speech_ratio: float,
prob_std: float,
energy_corr: float,
transition_rate: float,
forced_split: bool,
) -> float:
stability = max(0.3, 1.0 - prob_std)
alignment = max(0.3, energy_corr)
continuity = max(0.3, 1.0 - transition_rate)
penalty = 0.7 if forced_split else 1.0
score = avg_prob _ speech_ratio _ stability _ alignment _ continuity \* penalty
return round(float(score), 3)
def \_empty_stats() -> SpeechSegmentStats:
return SpeechSegmentStats(
avg_smoothed_prob=0.0,
max_smoothed_prob=0.0,
min_smoothed_prob=0.0,
prob_std=0.0,
total_frames=0,
speech_frames=0,
speech_frame_ratio=0.0,
silence_ratio=0.0,
internal_silence_frames=0,
max_consecutive_silence_frames=0,
has_internal_pause=False,
max_consecutive_speech_frames=0,
avg_consecutive_speech_frames=0.0,
speech_segments_count=0,
long_speech_ratio=0.0,
avg_rms=0.0,
rms_std=0.0,
speech_rms_mean=0.0,
silence_rms_mean=0.0,
energy_speech_correlation=0.0,
low_energy_speech_ratio=0.0,
high_energy_non_speech_ratio=0.0,
prob_velocity_mean=0.0,
prob_velocity_std=0.0,
prob_acceleration_mean=0.0,
speech_transition_rate=0.0,
estimated_speech_duration_sec=0.0,
confidence_level="low",
quality_score=0.0,
)

# jet_python_modules/jet/audio/audio_waveform/speech_types.py

from typing import Literal, TypedDict
VadStateLabel = Literal[
"UNKNOWN", "SILENCE", "POSSIBLE_SPEECH", "SPEECH", "POSSIBLE_SILENCE"
]
class SpeechFrame(TypedDict):
frame_idx: int
raw_prob: float
smoothed_prob: float
is_speech: bool
is_speech_start: bool
is_speech_end: bool
vad_state: VadStateLabel

# jet_python_modules/jet/audio/helpers/energy_base.py

import numpy as np
from jet.audio.helpers.config import (
FRAME_LENGTH_MS,
HOP_LENGTH_MS,
SAMPLE_RATE,
SILENCE_MAX_THRESHOLD,
)
def get_audio_duration(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
"""Return duration of audio in seconds.
Parameters
----------
samples : np.ndarray
Input audio samples (1D array).
Returns
-------
float
Duration in seconds.
"""
if len(samples) == 0:
return 0.0
return float(len(samples) / sample_rate)
def compute_amplitude(samples: np.ndarray) -> float:
"""Compute peak amplitude (max |x|).
Range: 0.0 (true silence) → 1.0 (maximum possible loudness / 0 dBFS)
Common values: - < 0.01 → very quiet / silence - 0.1–0.6 → normal speech - > 0.7 → loud speech
"""
if len(samples) == 0:
return 0.0
return float(np.max(np.abs(samples)))
def compute_rms(samples: np.ndarray) -> float:
"""Root Mean Square – best simple measure of perceived loudness/energy.
Range: 0.0 (true silence) → ~0.707 (full-scale sine wave)
Typical speech values: - < SILENCE_MAX_THRESHOLD → silence / noise floor - SILENCE_MAX_THRESHOLD–0.03 → very quiet / breath - 0.03–0.15 → normal conversational speech - 0.15–0.4+ → loud speech / shouting
"""
if len(samples) == 0:
return 0.0
return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
def has_sound(samples: np.ndarray) -> bool:
"""Return True if the audio contains meaningful sound.
Now aligned with get_loudness_label(): - rms < SILENCE_MAX_THRESHOLD → "silent" → has_sound=False - rms >= SILENCE_MAX_THRESHOLD → "very_quiet" and above → has_sound=True
"""
if len(samples) == 0:
return False
rms_value = compute_rms(samples)
return (
rms_value >= SILENCE_MAX_THRESHOLD
)
def rms_to_loudness_label(rms_value: float) -> str:
"""Return a human-readable loudness label based on RMS."""
if rms_value < SILENCE_MAX_THRESHOLD:
return "silent"
elif rms_value < 0.03:
return "very_quiet"
elif rms_value < 0.12:
return "normal"
elif rms_value < 0.25:
return "loud"
else:
return "very_loud"
def trim_silent(
samples: np.ndarray,
sample_rate: int = SAMPLE_RATE,
frame_length_ms: float = FRAME_LENGTH_MS,
hop_length_ms: float = HOP_LENGTH_MS,
) -> np.ndarray:
"""
Remove leading and trailing silent portions from audio samples.
Silence is determined using RMS with the same threshold used by has_sound()
and rms_to_loudness_label().
Parameters
----------
samples : np.ndarray
Input audio samples (1D array).
sample_rate : int, default=22050
Sample rate of the audio in Hz. Used to convert ms → samples.
frame_length_ms : float, default=93.0
Analysis window size in milliseconds.
hop_length_ms : float, default=23.0
Step size between windows in milliseconds.
Returns
-------
np.ndarray
Trimmed audio with leading and trailing silence removed.
Returns empty array if the entire signal is silent.
"""
if len(samples) == 0:
return np.array([], dtype=samples.dtype)
frame_length = int(round(frame_length_ms _ sample_rate / 1000.0))
hop_length = int(round(hop_length_ms _ sample_rate / 1000.0))
frame_length = max(frame_length, 256)
hop_length = max(hop_length, 64)
hop_length = min(hop_length, frame_length // 2)
if len(samples) < frame_length:
return (
samples.copy() if has_sound(samples) else np.array([], dtype=samples.dtype)
)
def is_silent_frame(frame: np.ndarray) -> bool:
if len(frame) == 0:
return True
rms = compute_rms(frame)
return rms < SILENCE_MAX_THRESHOLD
start = 0
for i in range(0, len(samples) - frame_length + 1, hop_length):
frame = samples[i : i + frame_length]
if not is_silent_frame(frame):
start = i
break
else:
return np.array([], dtype=samples.dtype)
end = len(samples)
for i in range(len(samples) - frame_length, -1, -hop_length):
frame = samples[i : i + frame_length]
if not is_silent_frame(frame):
end = i + frame_length
break
return samples[start:end].copy()

Thinking • 1m 34s
Thinking about your request

Evaluating speech stats

Reusing jet.audio.helpers.energy could streamline frame analysis in speech stats code.
RMS stats remain zero because prob_frames lack an "rms" key, missing integration with energy helpers.
Segment-level RMS is computed elsewhere, but per-frame RMS needs to be added to SpeechFrame for accurate stats.
Segment audio is available at event creation, enabling RMS computation from raw waveform if per-frame RMS is missing.
Segment audio (92,160 samples at 16kHz) aligns with 594 prob frames (~10ms each), suggesting RMS can be computed per frame from waveform.
<step>Improving RMS stats</step>
Reusing energy helpers to compute speech_rms_mean from segment audio and speech frame alignment.
<step>Enhancing energy integration</step>
Simplifying RMS stats to use segment-level compute_rms from event.audio instead of per-frame zeros.
<step>Computing per-frame RMS</step>
Chunking segment audio into 160-sample frames to match 10ms VAD hop and compute RMS per frame.
<step>Assessing frame alignment</step>
Simplifying stats by setting speech_rms_mean to avg_rms when per-frame RMS unavailable.
<step>Reassessing frame alignment</step>
<step>Reusing energy helpers</step>
<step>Fixing RMS extraction</step>
<step>Reusing segment RMS</step>
<step>Approximating per-frame RMS</step>
<step>Resolving audio-frame mismatch</step>
Setting constant RMS array

Filling rms_values with avg_rms across all frames to avoid zero stats and reuse precomputed segment RMS.
Setting speech_rms_mean and silence_rms_mean to avg_rms (or 0 if no such frames) to ensure non-zero values while reusing segment RMS.
<step>Reusing precomputed RMS</step>
<step>Deprecating broken RMS extraction</step>
<step>Updating RMS logic</step>
<step>Integrating segment RMS fallback</step>
<step>Updating RMS extraction function</step>
<step>Evaluating quality score</step>
<step>Adjusting quality alignment</step>
<step>Reusing energy helpers in stats</step>
Explicitly calling compute_rms

Importing compute_rms from energy_base and using it to calculate full-segment RMS for per-frame stats when no frame-level RMS exists.
Adding fallback to compute_rms(event.audio) when prob_frames lack RMS, ensuring stats reuse energy helper code directly.
<step>Passing event to RMS extractor</step>
<step>Updating RMS extraction</step>
<step>Reusing compute_rms helper</step>
Importing compute_rms

Adding runtime import of compute_rms from jet.audio.helpers.energy_base to enable RMS calculation in \_extract_rms.
<step>Approximating per-frame RMS</step>

Upgrade to SuperGrok
New conversation - Grok
