# run_speaker_similarity.py
from jet.audio.speech.pyannote.speaker_similarity import SpeakerSimilarity
from jet.logger import logger

# ── File paths ─────────────────────────────────────────────────────────────────
refs_son = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0001/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0002/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0003/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0004/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0005/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0006/sound.wav",
]
refs_mom = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0007/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0008/sound.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0009/sound.wav",
]

sim = SpeakerSimilarity()

# File paths
speaker_file1 = refs_son[0]
speaker_file2 = refs_son[1]
score_same = sim.similarity(speaker_file1, speaker_file2)
logger.success(f"Same speaker: {score_same:.4f}")     # expect 0.7–0.95+

# File paths
speaker_file1 = refs_mom[0]
speaker_file2 = refs_mom[1]
score_same = sim.similarity(speaker_file1, speaker_file2)
logger.success(f"Same speaker: {score_same:.4f}")     # expect 0.7–0.95+

# File paths
speaker_file1 = refs_son[0]
speaker_file2 = refs_mom[0]
score_diff = sim.similarity(speaker_file1, speaker_file2)
logger.success(f"Different: {score_diff:.4f}")        # expect < 0.5 typically

# # Numpy waveforms (e.g. from soundfile, torchaudio, librosa)
# import soundfile as sf
# import numpy as np
# wave_a, sr_a = sf.read("clipA.wav")
# wave_b, sr_b = sf.read("clipB.wav")

# # Ensure mono
# if wave_a.ndim > 1: wave_a = np.mean(wave_a, axis=1)
# if wave_b.ndim > 1: wave_b = np.mean(wave_b, axis=1)

# score = sim.similarity(wave_a, wave_b)
# print(f"Similarity from arrays: {score:.4f}")


# ── Cluster all segments ─────────────────────────────────────────────────────
logger.info("Clustering speakers")

# You can tune the threshold here (0.75–0.82 is a common range)
threshold = 0.78
# Combine all segments
# all_segments = [*refs_son, *refs_mom]
all_segments = [refs_son[0], refs_son[1]]

# Window step 1: Compute similarity for every consecutive segment pair
for i in range(len(all_segments) - 1):
    speaker_file1 = all_segments[i]
    speaker_file2 = all_segments[i + 1]
    score = sim.similarity(speaker_file1, speaker_file2)
    logger.success(f"Window step {i}: segment_{i+1:04d} vs segment_{i+2:04d} similarity = {score:.4f}")

# labels, _ = sim.assign_speaker_labels(
#     inputs=all_segments,
#     threshold=threshold
# )

# # ── Print nice table ─────────────────────────────────────────────────────────
# table = Table(title=f"Speaker Clustering Results (threshold = {threshold})")
# table.add_column("Index", justify="right", style="dim")
# table.add_column("Segment", style="cyan")
# table.add_column("Speaker Label", style="magenta bold", justify="center")
# table.add_column("File name")

# for i, (label, path) in enumerate(zip(labels, all_segments)):
#     table.add_row(
#         f"{i}",
#         f"segment_{i+1:04d}",
#         f"{label}",
#         Path(path).name,
#     )

# console.print(table)

# # Show distribution
# counts = Counter(labels)
# console.print("\n[bold]Speaker distribution:[/bold]")
# for label, count in sorted(counts.items()):
#     console.print(f"  Speaker {label}: {count} segments")
