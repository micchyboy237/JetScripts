from jet.audio.audio_search import AudioSegmentDatabase
from jet.audio.utils import resolve_audio_paths


if __name__ == "__main__":
    db = AudioSegmentDatabase(persist_dir="./my_audio_db")
    
    # Example 1: Index some audio files (run once)
    audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/segments"
    audio_files = resolve_audio_paths(audio_dir, recursive=True)
    for file in audio_files:
        db.add_segments_from_file(file, segment_duration_sec=30.0, overlap_sec=10.0)
    
    # Example 2: Search with a query file
    query_path = "path/to/query_segment.wav"
    results = db.search_similar(query_path, top_k=10)
    db.print_results(results)
    
    # Example 3: Search with raw audio bytes (e.g., from API upload)
    with open("path/to/query_segment.wav", "rb") as f:
        query_bytes = f.read()
    
    results_bytes = db.search_similar(query_bytes, top_k=5)
    db.print_results(results_bytes)