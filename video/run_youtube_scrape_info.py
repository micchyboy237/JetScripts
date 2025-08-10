import asyncio
import json
import os
import shutil
from jet.file.utils import save_file
from jet.utils.text import format_sub_dir
from jet.video.utils import download_audio
from jet.video.youtube.youtube_info_extractor import YoutubeInfoExtractor
from jet.video.youtube.youtube_scrape_info import find_audio, transcribe_youtube_video_info_and_chapters

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main():
    audio_format = 'mp3'
    audio_dir = f'{OUTPUT_DIR}'
    video_url = 'https://www.youtube.com/watch?v=7qr6DK6P0uQ'

    print(f"Extracting video info from:\n{video_url}")
    extractor = YoutubeInfoExtractor(video_url)
    info = await extractor.extract_info()
    formatted_title = format_sub_dir(info['title'])
    audio_dir = f"{audio_dir}/{formatted_title}"
    info_file_name = f"{audio_dir}/info.json"

    save_file(info, info_file_name)

    print(f"Info\n{json.dumps(info, indent=2)}")
    audio_paths = find_audio(audio_dir)
    print("audio_paths", audio_paths)

    if not audio_paths:
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = download_audio(video_url, audio_dir, audio_format)
        print(f"Downloaded audio file to:\n{audio_path}")
    else:
        audio_path = audio_paths[0]
        print(f"Reuse existing audio file:\n{audio_path}")

    print(f"Transcribing audio")
    transcriptions_output_file = f"{audio_dir}/transcriptions.json"
    for transcription in transcribe_youtube_video_info_and_chapters(video_url, audio_path, info, audio_dir):
        with open(transcriptions_output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(transcription) + '\n')

if __name__ == '__main__':
    asyncio.run(main())
