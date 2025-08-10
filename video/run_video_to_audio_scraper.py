from jet.video.video_to_audio_scraper import convert_video_to_audio


if __name__ == '__main__':
    video_path = '/Users/jethroestrada/Desktop/External_Projects/GPT/xturing-jet-examples/data/scrapers/test/initial-interview-1.mov'
    converted_audio_path = convert_video_to_audio(video_path)
    print(
        f"Converted video file {video_path} to audio file {converted_audio_path}")
