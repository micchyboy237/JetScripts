import asyncio
import os
import shutil
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry
from jet.video.youtube.youtube_playlist_extractor import YoutubePlaylistExtractor, transcribe_youtube_videos_playlist

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main():
    playlist_ids = [
        {
            "id": "PLPcB0_P-Zlj5NT6ukfitQyncIy4Vzk4L0",
            "title": "Can't Buy Me Love (Trending Scenes)",
        }
    ]
    model_size = 'small'
    model = WhisperModelRegistry.load_model(model_size)
    for item in playlist_ids:
        playlist_id = item['id']
        playlist_title = item['title']
        playlist_title = "_".join(playlist_title.split()).lower()
        playlist_title = f"playlist_{playlist_title}"
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
        extractor = YoutubePlaylistExtractor(playlist_url)
        video_ids = await extractor.extract_video_ids()  # Add await here

        transcribe_youtube_videos_playlist(
            model, video_ids, playlist_title, output_dir=OUTPUT_DIR)
        print(f"Done transcribing videos in playlist: {playlist_id}")

if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function
