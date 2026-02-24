import shutil
from pathlib import Path

from speechbrain.inference.ASR import EncoderDecoderASR

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech",
    savedir=str(OUTPUT_DIR / "pretrained_model"),
)
audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"
results = asr_model.transcribe_file(audio_file)
print(f"Results:\n{results}")
