from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from videodb import SceneExtractionType
from videodb import Segmenter
from videodb import connect
from videodb import play_stream
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/multi_modal_videorag_videodb.ipynb
" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# Multimodal RAG with VideoDB

### RAG: Multimodal Search on Videos and Stream Video Results 📺

Constructing a RAG pipeline for text is relatively straightforward, thanks to the tools developed for parsing, indexing, and retrieving text data. 

However, adapting RAG models for video content presents a greater challenge. Videos combine visual, auditory, and textual elements, requiring more processing power and sophisticated video pipelines.

> [VideoDB](https://videodb.io) is a serverless database designed to streamline the storage, search, editing, and streaming of video content. VideoDB offers random access to sequential video data by building indexes and developing interfaces for querying and browsing video content. Learn more at [docs.videodb.io](https://docs.videodb.io).

To build a truly Multimodal search for Videos, you need to work with different modalities of a video like Spoken Content, Visual.

In this notebook, we will develop a multimodal RAG for video using VideoDB and Llama-Index ✨.

![](https://raw.githubusercontent.com/video-db/videodb-cookbook-assets/main/images/guides/multimodal_llama_index_1.png)

&nbsp;
## 🛠️️ Setup 

---

### 🔑 Requirements

To connect to VideoDB, simply get the API key and create a connection. This can be done by setting the `VIDEO_DB_API_KEY` environment variable. You can get it from 👉🏼 [VideoDB Console](https://console.videodb.io). ( Free for first 50 uploads, **No credit card required!** )

# Get your `OPENAI_API_KEY` from OllamaFunctionCalling platform for `llama_index` response synthesizer.

# <!-- > Set the `OPENAI_API_KEY` & `VIDEO_DB_API_KEY` environment variable with your API keys. -->
"""
logger.info("# Multimodal RAG with VideoDB")


os.environ["VIDEO_DB_API_KEY"] = ""
# os.environ["OPENAI_API_KEY"] = ""

"""
### 📦 Installing Dependencies

To get started, we'll need to install the following packages:

- `llama-index`
- `videodb`
"""
logger.info("### 📦 Installing Dependencies")

# %pip install videodb
# %pip install llama-index

"""
## 🛠 Building Multimodal RAG

---

### 📋 Step 1: Connect to VideoDB and Upload Video

Let's upload a our video file first.

You can use any `public url`, `Youtube link` or `local file` on your system. 

> ✨ First 50 uploads are free!
"""
logger.info("## 🛠 Building Multimodal RAG")


conn = connect()
coll = conn.get_collection()

logger.debug("Uploading Video")
video = conn.upload(url="https://www.youtube.com/watch?v=libKVRa01L8")
logger.debug(f"Video uploaded with ID: {video.id}")

"""
> * `coll = conn.get_collection()` : Returns default collection object.
> * `coll.get_videos()` : Returns list of all the videos in a collections.
> * `coll.get_video(video_id)`: Returns Video object from given`video_id`.

### 📸🗣️ Step 2: Extract Scenes from Video

First, we need to extract scenes from the video and then use vLLM to obtain a description of each scene.

To learn more about Scene Extraction options, explore the following guides:
- [Scene Extraction Options Guide](https://github.com/video-db/videodb-cookbook/blob/main/guides/scene-index/playground_scene_extraction.ipynb) delves deeper into the various options available for scene extraction within Scene Index. It covers advanced settings, customization features, and tips for optimizing scene extraction based on different needs and preferences.
"""
logger.info("### 📸🗣️ Step 2: Extract Scenes from Video")



index_id = video.index_scenes(
    extraction_type=SceneExtractionType.time_based,
    extraction_config={"time": 2, "select_frames": ["first", "last"]},
    prompt="Describe the scene in detail",
)
video.get_scene_index(index_id)

logger.debug(f"Scene Extraction successful with ID: {index_id}")

"""
### ✨ Step 3 : Incorporating VideoDB in your existing Llamaindex RAG Pipeline
---

To develop a thorough multimodal search for videos, you need to handle different video modalities, including spoken content and visual elements.

You can retrieve all Transcript Nodes and Visual Nodes of a video using VideoDB and then incorporate them into your LlamaIndex pipeline.

#### 🗣 Fetching Transcript Nodes

You can fetch transcript nodes using `Video.get_transcript()`

To configure the segmenter, use the `segmenter` and `length` arguments.

Possible values for segmenter are:
- `Segmenter.time`: Segments the video based on the specified `length` in seconds.
- `Segmenter.word`: Segments the video based on the word count specified by `length`
"""
logger.info("### ✨ Step 3 : Incorporating VideoDB in your existing Llamaindex RAG Pipeline")



nodes_transcript_raw = video.get_transcript(
    segmenter=Segmenter.time, length=60
)

nodes_transcript = [
    TextNode(
        text=node["text"],
        metadata={key: value for key, value in node.items() if key != "text"},
    )
    for node in nodes_transcript_raw
]

"""
#### 📸 Fetching Scene Nodes
"""
logger.info("#### 📸 Fetching Scene Nodes")

scenes = video.get_scene_index(index_id)

nodes_scenes = [
    TextNode(
        text=node["description"],
        metadata={
            key: value for key, value in node.items() if key != "description"
        },
    )
    for node in scenes
]

"""
### 🔄 Simple RAG Pipeline with Transcript + Scene Nodes

We index both our Transcript Nodes and Scene Node

🔍✨ For simplicity, we are using a basic RAG pipeline. However, you can integrate more advanced LlamaIndex RAG pipelines here for better results.
"""
logger.info("### 🔄 Simple RAG Pipeline with Transcript + Scene Nodes")


index = VectorStoreIndex(nodes_scenes + nodes_transcript)
q = index.as_query_engine()

"""
#### ️💬️ Viewing the result : Text
"""
logger.info("#### ️💬️ Viewing the result : Text")

res = q.query(
    "Show me where the narrator discusses the formation of the solar system and visualize the milky way galaxy"
)
logger.debug(res)

"""
#### 🎥 Viewing the result : Video Clip

Our nodes' metadata includes `start` and `end` fields, which represent the start and end times relative to the beginning of the video.

Using this information from the relevant nodes, we can create Video Clips corresponding to these nodes.
"""
logger.info("#### 🎥 Viewing the result : Video Clip")



def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for interval in intervals[1:]:
        if interval[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)
    return merged


relevant_timestamps = [
    [node.metadata["start"], node.metadata["end"]] for node in res.source_nodes
]

stream_url = video.generate_stream(merge_intervals(relevant_timestamps))
play_stream(stream_url)

"""
## 🏃‍♂️ Next Steps
---

In this guide, we built a Simple Multimodal RAG for Videos Using VideoDB, Llamaindex, and OllamaFunctionCalling

You can optimize the pipeline by incorporating more advanced techniques like
- Build a Search on Video Collection
- Optimize Query Transformation
- More methods to combine retrieved nodes from different modalities
- Experiment with Different RAG pipelines like Knowledge Graph


To learn more about Scene Index, explore the following guides:

- [Quickstart Guide](https://github.com/video-db/videodb-cookbook/blob/main/quickstart/Scene%20Index%20QuickStart.ipynb) 
- [Scene Extraction Options](https://github.com/video-db/videodb-cookbook/blob/main/guides/scene-index/playground_scene_extraction.ipynb)
- [Advanced Visual Search](https://github.com/video-db/videodb-cookbook/blob/main/guides/scene-index/advanced_visual_search.ipynb)
- [Custom Annotation Pipelines](https://github.com/video-db/videodb-cookbook/blob/main/guides/scene-index/custom_annotations.ipynb)

## 👨‍👩‍👧‍👦 Support & Community
---

If you have any questions or feedback. Feel free to reach out to us 🙌🏼

* [Discord](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdiscord.gg%2Fpy9P639jGz)
* [GitHub](https://github.com/video-db)
* [Email](mailto:ashu@videodb.io)
"""
logger.info("## 🏃‍♂️ Next Steps")

logger.info("\n\n[DONE]", bright=True)