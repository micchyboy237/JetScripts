from pathlib import Path
import shutil

import numpy as np
import scipy.io.wavfile as wavfile
import noisereduce as nr
from rich import print as rprint

def denoise_audio(
    input_path: Path,
    output_path: Path,
    noise_sample_length_seconds: float = 1.0,
    prop_decrease: float = 0.8,
    stationary: bool = False,
) -> None:
    """
    Denoise a WAV file using spectral gating with noisereduce.

    Args:
        input_path: Path to input WAV (assumed 16kHz mono int16 for your use case).
        output_path: Path to save denoised WAV.
        noise_sample_length_seconds: Length (in seconds) of initial noise sample to estimate profile.
                                     Use longer for better stationary noise estimation.
        prop_decrease: Proportion of noise to reduce (0.0 = none, 1.0 = aggressive).
        stationary: If True, assume noise is stationary (faster, slightly better for constant noise).
    """
    rate, audio = wavfile.read(input_path)
    
    if rate != 16000:
        raise ValueError(f"Expected 16kHz audio, got {rate}Hz")
    
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]  # Take first channel if stereo

    # Estimate noise from initial segment (common for speech recordings with leading noise/silence)
    noise_samples = int(rate * noise_sample_length_seconds)
    noise_clip = audio[:noise_samples]
    
    rprint(f"[bold green]Denoising[/bold green] {input_path.name} using initial {noise_sample_length_seconds}s as noise profile...")
    
    # Perform reduction with progress bar
    denoised = nr.reduce_noise(
        y=audio,
        y_noise=noise_clip,
        sr=rate,
        prop_decrease=prop_decrease,
        stationary=stationary,
        use_tqdm=True,
        n_fft=512,  # Good default for speech
    )
    
    # Clip to int16 range and convert back
    denoised = np.clip(denoised, -32768, 32767).astype(np.int16)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, rate, denoised)
    rprint(f"[bold green]Denoised audio saved to:[/bold green] {output_path}")

# Example usage matching your script
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

audio_path = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0006/sound.wav")
denoised_path = OUTPUT_DIR / "segment_0006_denoised.wav"

denoise_audio(audio_path, denoised_path, noise_sample_length_seconds=1.0, prop_decrease=0.8)