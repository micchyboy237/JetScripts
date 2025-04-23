import os
import yt_dlp
from pydub import AudioSegment
from faster_whisper import WhisperModel


# Function to download audio from YouTube video
def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioquality': 1,
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegAudioConvertor',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        audio_file = f"{info['title']}.mp3"
        return audio_file


# Function to transcribe an audio chunk (buffer)
def transcribe_chunk(model, audio_chunk):
    segments, info = model.transcribe(audio_chunk, beam_size=5)
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


# Function to transcribe the audio in buffers (chunks)
def transcribe_in_buffers(audio_file, chunk_duration_ms=30000):
    # Load the audio file
    audio = AudioSegment.from_mp3(audio_file)

    # Initialize Whisper model (use "int8" or "float16" based on your preference)
    model = WhisperModel("small", compute_type="int8")

    # Process audio in chunks
    total_duration = len(audio)
    print(f"Total audio duration: {total_duration / 1000} seconds")

    # Transcribe each chunk
    for start_ms in range(0, total_duration, chunk_duration_ms):
        end_ms = min(start_ms + chunk_duration_ms, total_duration)
        audio_chunk = audio[start_ms:end_ms]

        # Save the chunk as a temporary file (in-memory buffering or disk saving)
        chunk_file = "temp_chunk.wav"
        audio_chunk.export(chunk_file, format="wav")

        # Transcribe the chunk
        transcribe_chunk(model, chunk_file)

        # Optionally clean up
        os.remove(chunk_file)


# Main function to run the process
def main(youtube_url):
    print("Downloading audio from YouTube...")
    audio_file = download_audio(youtube_url)
    print(f"Audio downloaded as {audio_file}")

    print("Transcribing audio in buffers...")
    transcribe_in_buffers(audio_file)

    # Clean up the audio file after transcription
    os.remove(audio_file)
    print(f"Deleted {audio_file} after transcription.")


# Run the script
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example link
    main(youtube_url)
