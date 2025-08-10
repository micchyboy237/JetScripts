import os
import shutil
from jet.video.youtube.youtube_info_extractor import transcribe_youtube_videos_info

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    video_ids = ["Unzc731iCUY"]  # YOUTUBE_TAGALOG_TRANSLATIONS

    transcribe_youtube_videos_info(video_ids, OUTPUT_DIR)
