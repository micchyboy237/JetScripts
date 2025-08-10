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
    audio_dir = f'{OUTPUT_DIR}'
    video_url = 'https://www.youtube.com/watch?v=7qr6DK6P0uQ'

    print(f"Extracting video info from:\n{video_url}")
    extractor = YoutubeInfoExtractor(video_url)
    info = await extractor.extract_info()
    formatted_title = format_sub_dir(info['title'])
    audio_dir = f"{audio_dir}/{formatted_title}"
    info_file_name = f"{audio_dir}/info.json"

    save_file(info, info_file_name)


if __name__ == '__main__':
    asyncio.run(main())
