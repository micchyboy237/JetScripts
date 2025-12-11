# instantiate the pipeline
import os
import shutil
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
import torch
import torchaudio
from pathlib import Path
import json
import numpy as np

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.getenv("HF_TOKEN")
)

# send pipeline to GPU/MPS/CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
pipeline.to(device)

# run the pipeline on an audio file
audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251212_041845.wav"
diarization: DiarizeOutput = pipeline(audio_file, num_speakers=2)


def save_diarize_output(output: DiarizeOutput, out_dir: str | Path, uri: str = "audio"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed: use torchaudio.save instead of deprecated save_waveform
    audio = Audio(sample_rate=16000, mono="downmix")

    # ------------------------------------------------------------------
    # Global files
    # ------------------------------------------------------------------
    (out_dir / f"{uri}.rttm").write_text(output.speaker_diarization.to_rttm())
    (out_dir / f"{uri}_exclusive.rttm").write_text(output.exclusive_speaker_diarization.to_rttm())

    json_data = output.serialize()
    if output.speaker_embeddings is not None:
        json_data["speaker_embeddings"] = {
            spk: emb.tolist()
            for spk, emb in zip(output.speaker_diarization.labels(), output.speaker_embeddings)
        }
        json_data["embedding_dim"] = output.speaker_embeddings.shape[1]
        np.save(out_dir / f"{uri}_embeddings.npy", output.speaker_embeddings)

    np.savetxt(out_dir / f"{uri}_speaker_order.txt", output.speaker_diarization.labels(), fmt="%s")
    (out_dir / f"{uri}.json").write_text(json.dumps(json_data, indent=2))

    # ------------------------------------------------------------------
    # Per-turn segments
    # ------------------------------------------------------------------
    segments_dir = out_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    manifest = []
    for idx, (turn, _, speaker) in enumerate(output.speaker_diarization.itertracks(yield_label=True)):
        seg_dir = segments_dir / f"segment_{idx:04d}"
        seg_dir.mkdir(exist_ok=True)

        # Extract waveform (returns torch.Tensor + sample rate)
        waveform, sample_rate = audio.crop(Path(audio_file), turn)

        # Save using torchaudio (official current method)
        wav_path = seg_dir / "audio.wav"
        torchaudio.save(wav_path, waveform, sample_rate)

        # Per-segment metadata
        turn_info = {
            "index": idx,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
            "speaker": speaker,
            "wav": str(wav_path.relative_to(out_dir)),
        }
        (seg_dir / "turn.json").write_text(json.dumps(turn_info, indent=2))

        # Optional single-line RTTM
        rttm_line = f"SPEAKER {uri} 1 {turn.start:.3f} {turn.duration:.3f} <NA> <NA> {speaker} <NA>\n"
        (seg_dir / "turn.rttm").write_text(rttm_line)

        manifest.append(turn_info)

    (out_dir / "segments_manifest.json").write_text(json.dumps(manifest, indent=2))


# ----------------------------------------------------------------------
# Run saver
# ----------------------------------------------------------------------
output_dir = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Diarization output written under: {output_dir.resolve()}")
save_diarize_output(diarization, output_dir, uri=Path(audio_file).stem)