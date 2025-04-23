import os
from pydub import AudioSegment
from faster_whisper import WhisperModel


# Global variable to track the end time of the last transcribed segment
last_transcribed_end_time = 0.0


# Function to transcribe an audio chunk (buffer)
def transcribe_chunk(model, audio_chunk, output_file, time_offset=0):
    global last_transcribed_end_time
    segments, info = model.transcribe(audio_chunk, beam_size=5)
    with open(output_file, "a") as f:  # Append mode to avoid overwriting
        for segment in segments:
            # Adjust timestamps with the chunk's time offset
            start_time = segment.start + time_offset
            end_time = segment.end + time_offset

            # Skip segments that are within the overlap region of the previous chunk
            if start_time < last_transcribed_end_time:
                continue

            transcription = f"[{start_time:.2f}s -> {end_time:.2f}s] {segment.text}\n"
            print(transcription, end="")  # Print to console
            f.write(transcription)  # Save to file

            # Update the last transcribed end time
            last_transcribed_end_time = end_time


# Function to transcribe the audio in buffers (chunks)
def transcribe_in_buffers(audio_file, output_dir, chunk_duration_ms=30000, overlap_ms=1000):
    global last_transcribed_end_time
    last_transcribed_end_time = 0.0  # Reset for each new audio file

    # Load the audio file
    audio = AudioSegment.from_file(audio_file)

    # Get the original file extension (without the dot) and format
    original_extension = os.path.splitext(audio_file)[1].lower().lstrip(".")
    if original_extension == "mp3":
        export_format = "mp3"
    elif original_extension == "wav":
        export_format = "wav"
    else:
        export_format = "wav"  # Fallback to WAV for unsupported formats
        print(
            f"Unsupported input format '{original_extension}'. Using WAV for chunks.")

    # Initialize Whisper model (use "int8" or "float16" based on your preference)
    model = WhisperModel("small", compute_type="int8")

    # Process audio in chunks
    total_duration = len(audio)
    print(f"Total audio duration: {total_duration / 1000} seconds")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define output file path
    output_file = os.path.join(output_dir, "transcription.txt")
    # Ensure the output file is empty before starting
    if os.path.exists(output_file):
        os.remove(output_file)

    # Transcribe each chunk
    step_size = chunk_duration_ms - overlap_ms  # Adjust step size for overlap
    for start_ms in range(0, total_duration, step_size):
        end_ms = min(start_ms + chunk_duration_ms, total_duration)
        audio_chunk = audio[start_ms:end_ms]

        # Save the chunk as a temporary file in output_dir with original format
        chunk_file = os.path.join(
            output_dir, f"temp_chunk.{original_extension}")
        audio_chunk.export(chunk_file, format=export_format)

        # Calculate time offset for this chunk (in seconds)
        time_offset = start_ms / 1000.0

        # Transcribe the chunk
        transcribe_chunk(model, chunk_file, output_file,
                         time_offset=time_offset)

        # Clean up
        os.remove(chunk_file)


# Main function to run the process
def transcribe_file(audio_file: str, output_dir: str, *, chunk_duration_ms: int = 30000, overlap_ms: int = 1000, remove_audio: bool = False):
    print(f"Transcribing audio file: {audio_file}")

    print("Transcribing audio in buffers...")
    transcribe_in_buffers(audio_file, output_dir,
                          chunk_duration_ms, overlap_ms)

    # Clean up the audio file after transcription if the flag is set
    if remove_audio:
        os.remove(audio_file)
        print(f"Deleted {audio_file} after transcription.")
    else:
        print(
            f"Transcription complete. Original audio file '{audio_file}' was NOT removed.")


# Run the script
if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/agent/tts_Interviewer_20250423_231046_181704_Interviewer_Lets_begin_the_i.mp3"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/output"
    transcribe_file(audio_file, output_dir, chunk_duration_ms=30000,
                    overlap_ms=1000, remove_audio=False)
