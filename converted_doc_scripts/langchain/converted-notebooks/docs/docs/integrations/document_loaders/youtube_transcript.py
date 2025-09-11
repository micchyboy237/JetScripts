from jet.logger import logger
from langchain_community.document_loaders import GoogleApiClient, GoogleApiYoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from pathlib import Path
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# YouTube transcripts

>[YouTube](https://www.youtube.com/) is an online video sharing and social media platform created by Google.

This notebook covers how to load documents from `YouTube transcripts`.
"""
logger.info("# YouTube transcripts")


# %pip install --upgrade --quiet  youtube-transcript-api

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=QsYGlZkevEg", add_video_info=False
)

loader.load()

"""
### Add video info
"""
logger.info("### Add video info")

# %pip install --upgrade --quiet  pytube

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=QsYGlZkevEg", add_video_info=True
)
loader.load()

"""
### Add language preferences

Language param : It's a list of language codes in a descending priority, `en` by default.

translation param : It's a translate preference, you can translate available transcript to your preferred language.
"""
logger.info("### Add language preferences")

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=QsYGlZkevEg",
    add_video_info=True,
    language=["en", "id"],
    translation="en",
)
loader.load()

"""
### Get transcripts as timestamped chunks

Get one or more `Document` objects, each containing a chunk of the video transcript.  The length of the chunks, in seconds, may be specified.  Each chunk's metadata includes a URL of the video on YouTube, which will start the video at the beginning of the specific chunk.

`transcript_format` param:  One of the `langchain_community.document_loaders.youtube.TranscriptFormat` values.  In this case, `TranscriptFormat.CHUNKS`.

`chunk_size_seconds` param:  An integer number of video seconds to be represented by each chunk of transcript data.  Default is 120 seconds.
"""
logger.info("### Get transcripts as timestamped chunks")


loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=TKCMw0utiak",
    add_video_info=True,
    transcript_format=TranscriptFormat.CHUNKS,
    chunk_size_seconds=30,
)
logger.debug("\n\n".join(map(repr, loader.load())))

"""
## YouTube loader from Google Cloud

### Prerequisites

1. Create a Google Cloud project or use an existing project
1. Enable the [Youtube Api](https://console.cloud.google.com/apis/enableflow?apiid=youtube.googleapis.com&project=sixth-grammar-344520)
1. [Authorize credentials for desktop app](https://developers.google.com/drive/api/quickstart/python#authorize_credentials_for_a_desktop_application)
1. `pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib youtube-transcript-api`

### 🧑 Instructions for ingesting your Google Docs data
By default, the `GoogleDriveLoader` expects the `credentials.json` file to be `~/.credentials/credentials.json`, but this is configurable using the `credentials_file` keyword argument. Same thing with `token.json`. Note that `token.json` will be created automatically the first time you use the loader.

`GoogleApiYoutubeLoader` can load from a list of Google Docs document ids or a folder id. You can obtain your folder and document id from the URL:
Note depending on your set up, the `service_account_path` needs to be set up. See [here](https://developers.google.com/drive/api/v3/quickstart/python) for more details.
"""
logger.info("## YouTube loader from Google Cloud")



google_api_client = GoogleApiClient(credentials_path=Path("your_path_creds.json"))


youtube_loader_channel = GoogleApiYoutubeLoader(
    google_api_client=google_api_client,
    channel_name="Reducible",
    captions_language="en",
)


youtube_loader_ids = GoogleApiYoutubeLoader(
    google_api_client=google_api_client, video_ids=["TrdevFK_am4"], add_video_info=True
)

youtube_loader_channel.load()

logger.info("\n\n[DONE]", bright=True)