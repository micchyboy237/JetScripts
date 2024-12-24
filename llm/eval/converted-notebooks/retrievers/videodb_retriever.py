from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

### RAG: Multimodal Search on Videos and Stream Video Results 📺
# 
# Constructing a RAG pipeline for text is relatively straightforward, thanks to the tools developed for parsing, indexing, and retrieving text data. 
# 
# However, adapting RAG models for video content presents a greater challenge. Videos combine visual, auditory, and textual elements, requiring more processing power and sophisticated video pipelines.
# 
# While Large Language Models (LLMs) excel with text, they fall short in helping you consume or create video clips. `VideoDB` provides a sophisticated database abstraction for your MP4 files, enabling the use of LLMs on your video data. With VideoDB, you can not only analyze but also `instantly watch video streams` of your search results.
# 
# > [VideoDB](https://videodb.io) is a serverless database designed to streamline the storage, search, editing, and streaming of video content. VideoDB offers random access to sequential video data by building indexes and developing interfaces for querying and browsing video content. Learn more at [docs.videodb.io](https://docs.videodb.io).
# 
# 
# In this notebook, we introduce `VideoDBRetriever`, a tool specifically designed to simplify the creation of RAG pipelines for video content, without any hassle of dealing with complex video infrastructure.
# 
# ![](https://raw.githubusercontent.com/video-db/videodb-cookbook-assets/main/images/guides/multimodal_llama_index_2.png)

# &nbsp;
## 🛠️️ Setup 
# 
# ---

### 🔑 Requirements
# 
# To connect to VideoDB, simply get the API key and create a connection. This can be done by setting the `VIDEO_DB_API_KEY` environment variable. You can get it from 👉🏼 [VideoDB Console](https://console.videodb.io). ( Free for first 50 uploads, **No credit card required!** )
# 
# Get your `OPENAI_API_KEY` from Ollama platform for `llama_index` response synthesizer.
# 
# <!-- > Set the `OPENAI_API_KEY` & `VIDEO_DB_API_KEY` environment variable with your API keys. -->

import os

os.environ["VIDEO_DB_API_KEY"] = ""
# os.environ["OPENAI_API_KEY"] = ""

### 📦 Installing Dependencies
# 
# To get started, we'll need to install the following packages:
# 
# - `llama-index`
# - `llama-index-retrievers-videodb`
# - `videodb`

# %pip install videodb
# %pip install llama-index

# %pip install llama-index-retrievers-videodb

## 🛠 Using VideoDBRetriever to Build RAG for Single Video
# 
# ---

### 📋 Step 1: Connect to VideoDB and Ingest Data
# 
# Let's upload a our video file first.
# 
# You can use any `public url`, `Youtube link` or `local file` on your system. 
# 
# > ✨ First 50 uploads are free!

from videodb import connect

conn = connect()
coll = conn.create_collection(
    name="VideoDB Retrievers", description="VideoDB Retrievers"
)

print("Uploading Video")
video = coll.upload(url="https://www.youtube.com/watch?v=aRgP3n0XiMc")
print(f"Video uploaded with ID: {video.id}")

# > * `coll = conn.get_collection()` : Returns default collection object.
# > * `coll.get_videos()` : Returns list of all the videos in a collections.
# > * `coll.get_video(video_id)`: Returns Video object from given`video_id`.

### 🗣️ Step 2: Indexing & Search from Spoken Content

# Video can be viewed as data with different modalities. First, we will work with the `spoken content`.

#### 🗣️ Indexing Spoken Content

print("Indexing spoken content in Video...")
video.index_spoken_words()

#### 🗣️ Retrieving Relevant Nodes from Spoken Index

# We will use the `VideoDBRetriever` to retrieve relevant nodes from our indexed content. The video ID should be passed as a parameter, and the `index_type` should be set to `IndexType.spoken_word`.
# 
# You can configure the `score_threshold` and `result_threshold` after experimentation.

from llama_index.retrievers.videodb import VideoDBRetriever
from videodb import SearchType, IndexType

spoken_retriever = VideoDBRetriever(
    collection=coll.id,
    video=video.id,
    search_type=SearchType.semantic,
    index_type=IndexType.spoken_word,
    score_threshold=0.1,
)

spoken_query = "Nationwide exams"
nodes_spoken_index = spoken_retriever.retrieve(spoken_query)

#### 🗣️️️ Viewing the result : 💬 Text

# We will use the relevant nodes and synthesize the response using llamaindex

from llama_index.core import get_response_synthesizer

response_synthesizer = get_response_synthesizer()

response = response_synthesizer.synthesize(
    spoken_query, nodes=nodes_spoken_index
)
print(response)

#### 🗣️ Viewing the result : 🎥 Video Clip

# For each retrieved node that is relevant to the query, the `start` and `end` fields in the metadata represent the time interval covered by the node.
# 
# We will use VideoDB's Programmable Stream to generate a stream of relevant video clips based on the timestamps of these nodes.

from videodb import play_stream

results = [
    (node.metadata["start"], node.metadata["end"])
    for node in nodes_spoken_index
]

stream_link = video.generate_stream(results)
play_stream(stream_link)

### 📸️ Step3 : Index & Search from Visual Content

#### 📸 Indexing Visual Content
# To learn more about Scene Index, explore the following guides:
# 
# - [Quickstart Guide](https://github.com/video-db/videodb-cookbook/blob/main/quickstart/Scene%20Index%20QuickStart.ipynb) guide provides a step-by-step introduction to Scene Index. It's ideal for getting started quickly and understanding the primary functions.
# 
# - [Scene Extraction Options Guide](https://github.com/video-db/videodb-cookbook/blob/main/guides/scene-index/playground_scene_extraction.ipynb) delves deeper into the various options available for scene extraction within Scene Index. It covers advanced settings, customization features, and tips for optimizing scene extraction based on different needs and preferences.

from videodb import SceneExtractionType

print("Indexing Visual content in Video...")

index_id = video.index_scenes(
    extraction_type=SceneExtractionType.shot_based,
    extraction_config={"frame_count": 3},
    prompt="Describe the scene in detail",
)
video.get_scene_index(index_id)

print(f"Scene Index successful with ID: {index_id}")

#### 📸️ Retrieving Relevant Nodes from Scene Index

# Just like we used `VideoDBRetriever` for the spoken index, we will use it for the scene index. Here, we will need to set `index_type` to `IndexType.scene` and pass the `scene_index_id`

from llama_index.retrievers.videodb import VideoDBRetriever
from videodb import SearchType, IndexType


scene_retriever = VideoDBRetriever(
    collection=coll.id,
    video=video.id,
    search_type=SearchType.semantic,
    index_type=IndexType.scene,
    scene_index_id=index_id,
    score_threshold=0.1,
)

scene_query = "accident scenes"
nodes_scene_index = scene_retriever.retrieve(scene_query)

#### 📸️️️ Viewing the result : 💬 Text

from llama_index.core import get_response_synthesizer

response_synthesizer = get_response_synthesizer()

response = response_synthesizer.synthesize(
    scene_query, nodes=nodes_scene_index
)
print(response)

#### 📸 ️ Viewing the result : 🎥 Video Clip

from videodb import play_stream

results = [
    (node.metadata["start"], node.metadata["end"])
    for node in nodes_scene_index
]

stream_link = video.generate_stream(results)
play_stream(stream_link)

### 🛠️ Step4: Simple Multimodal RAG - Combining Results of Both modalities

# We want to unlock in multimodal queries in our video library like this: 
# 
# > 📸🗣️ "*Show me 1.Accident Scene 2.Discussion about nationwide exams*"
# 
# There are lots of way to do create a multimodal RAG, for the sake of simplicity we are choosing a simple approach:
# 
# 1. 🧩 **Query Transformation**: Divide query into two parts that can be used with respective scene and spoken indexes.
# 2. 🔎 **Finding Relevant nodes for each modality**: Using `VideoDBRetriever` find relevant nodes from Spoken Index and Scene Index 
# 3. ✏️ **Viewing the result : Text**: Use Relevant Nodes to sythesize a text reponse Integrating the results from both indexes for precise video segment identification. 
# 4. 🎥 **Viewing the result : Video Clip**: Integrating the results from both indexes for precise video segment identification. 
# 
# > To checkout more advanced multimodal techniques, checkout out [advnaced multimodal guides](https://docs.videodb.io/multimodal-guide-90)

#### 🧩 Query Transformation

from llama_index.llms.ollama import Ollama


def split_spoken_visual_query(query):
    transformation_prompt = """
    Divide the following query into two distinct parts: one for spoken content and one for visual content. The spoken content should refer to any narration, dialogue, or verbal explanations and The visual content should refer to any images, videos, or graphical representations. Format the response strictly as:\nSpoken: <spoken_query>\nVisual: <visual_query>\n\nQuery: {query}
    """
    prompt = transformation_prompt.format(query=query)
    response = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096).complete(prompt)
    divided_query = response.text.strip().split("\n")
    spoken_query = divided_query[0].replace("Spoken:", "").strip()
    scene_query = divided_query[1].replace("Visual:", "").strip()
    return spoken_query, scene_query


query = "Show me 1.Accident Scene 2.Discussion about nationwide exams "
spoken_query, scene_query = split_spoken_visual_query(query)
print("Query for Spoken retriever : ", spoken_query)
print("Query for Scene retriever : ", scene_query)

##### 🔎 Finding Relevant nodes for each modality

from videodb import SearchType, IndexType

spoken_retriever = VideoDBRetriever(
    collection=coll.id,
    video=video.id,
    search_type=SearchType.semantic,
    index_type=IndexType.spoken_word,
    score_threshold=0.1,
)

scene_retriever = VideoDBRetriever(
    collection=coll.id,
    video=video.id,
    search_type=SearchType.semantic,
    index_type=IndexType.scene,
    scene_index_id=index_id,
    score_threshold=0.1,
)

nodes_spoken_index = spoken_retriever.retrieve(spoken_query)

nodes_scene_index = scene_retriever.retrieve(scene_query)

#### ️💬️ Viewing the result : Text

response_synthesizer = get_response_synthesizer()

response = response_synthesizer.synthesize(
    query, nodes=nodes_scene_index + nodes_spoken_index
)
print(response)

#### 🎥 Viewing the result : Video Clip

# From each modality, we have retrieved results that are relevant to the query within that specific modality (semantic and scene/visual, in this case).
# 
# Each node has start and end fields in the metadata, which represent the time interval the node covers.
# 
# There are lots of way to sythesize there results, For now we will use a simple method : 
# 
# - `Union`: This method takes all the timestamps from every node, creating a comprehensive list that includes every relevant time, even if some timestamps appear in only one modality.
# 
# One of the other ways can be `Intersection`:
# 
# - `Intersection`: This method only includes timestamps that are present in every node, resulting in a smaller list with times that are universally relevant across all modalities.

from videodb import play_stream


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


results = [
    [node.metadata["start"], node.metadata["end"]]
    for node in nodes_spoken_index + nodes_scene_index
]
merged_results = merge_intervals(results)

stream_link = video.generate_stream(merged_results)
play_stream(stream_link)

## 🛠 Using VideoDBRetriever to Build RAG for Collection of Videos
# ---

### Adding More videos to our collection

video_2 = coll.upload(url="https://www.youtube.com/watch?v=kMRX3EA68g4")

#### 🗣️ Indexing Spoken Content

video_2.index_spoken_words()

#### 📸 Indexing Scenes

from videodb import SceneExtractionType

print("Indexing Visual content in Video...")

index_id = video_2.index_scenes(
    extraction_type=SceneExtractionType.shot_based,
    extraction_config={"frame_count": 3},
    prompt="Describe the scene in detail",
)
video_2.get_scene_index(index_id)

print(f"Scene Index successful with ID: {index_id}")

#### 🧩 Query Transformation

query = "Show me 1.Accident Scene 2.Kiara is speaking "
spoken_query, scene_query = split_spoken_visual_query(query)
print("Query for Spoken retriever : ", spoken_query)
print("Query for Scene retriever : ", scene_query)

#### 🔎 Finding relevant nodes

from videodb import SearchType, IndexType

spoken_retriever = VideoDBRetriever(
    collection=coll.id,
    search_type=SearchType.semantic,
    index_type=IndexType.spoken_word,
    score_threshold=0.2,
)

scene_retriever = VideoDBRetriever(
    collection=coll.id,
    search_type=SearchType.semantic,
    index_type=IndexType.scene,
    score_threshold=0.2,
)

nodes_spoken_index = spoken_retriever.retrieve(spoken_query)

nodes_scene_index = scene_retriever.retrieve(scene_query)

#### ️💬️ Viewing the result : Text

response_synthesizer = get_response_synthesizer()

response = response_synthesizer.synthesize(
    "What is kaira speaking. And tell me about accident scene",
    nodes=nodes_scene_index + nodes_spoken_index,
)
print(response)

#### 🎥 Viewing the result : Video Clip

# When working with an editing workflow involving multiple videos, we need to create a `Timeline` of `VideoAsset` and then compile them.

# ![](https://codaio.imgix.net/docs/_s5lUnUCIU/blobs/bl-n4vT_dFztl/e664f43dbd4da89c3a3bfc92e3224c8a188eb19d2d458bebe049e780f72506ca6b19421c7168205f7ad307187e73da60c73cdbb9a0ef3fec77cc711927ad26a29a92cd13691fa9375c231f1c006853bacf28e09b3bf0bbcb5f7b76462b354a180fb437ad?auto=format%2Ccompress&fit=max)

from videodb import connect, play_stream
from videodb.timeline import Timeline
from videodb.asset import VideoAsset

timeline = Timeline(conn)

for node_obj in nodes_scene_index + nodes_spoken_index:
    node = node_obj.node

    node_asset = VideoAsset(
        asset_id=node.metadata["video_id"],
        start=node.metadata["start"],
        end=node.metadata["end"],
    )

    timeline.add_inline(node_asset)

stream_url = timeline.generate_stream()
play_stream(stream_url)

# &nbsp;
## Configuring `VideoDBRetriever`
# ---
# 
### ⚙️ Retriever for only one Video
# You can pass the `id` of the video object to search in only that video. 
# ```python
# VideoDBRetriever(video="my_video.id")
# ```
# 
### ⚙️ Retriever for a set of Video/ Collection
# You can pass the `id` of the Collection to search in only that Collection. 
# ```python
# VideoDBRetriever(collection="my_coll.id")
# ```
# 
### ⚙️ Retriever for different type of Indexes
# ```python
# from videodb import IndexType
# spoken_word = VideoDBRetriever(index_type=IndexType.spoken_word)
# 
# scene_retriever = VideoDBRetriever(index_type=IndexType.scene, scene_index_id="my_index_id")
# ```
# 
### ⚙️ Configuring Search Type of Retriever 
# `search_type` determines the search method used to retrieve nodes against given query 
# ```python
# from videodb import SearchType, IndexType
# 
# keyword_spoken_search = VideoDBRetriever(
#     search_type=SearchType.keyword,
#     index_type=IndexType.spoken_word
# )
# 
# semantic_scene_search = VideoDBRetriever(
#     search_type=SearchType.semantic,
#     index_type=IndexType.spoken_word
# )
# ```
# 
### ⚙️ Configure threshold parameters  
# - `result_threshold`: is the threshold for number of results returned by retriever; the default value is `5`
# - `score_threshold`: only nodes with score higher than `score_threshold` will be returned by retriever; the default value is `0.2`  
# 
# ```python
# custom_retriever = VideoDBRetriever(result_threshold=2, score_threshold=0.5)
# ```

## ✨ Configuring Indexing and Chunking
# ---
# 
# In this example, we utilize the VideoDB's Indexing for video retrieval. However, you have the flexibility to load both Transcript and Scene Data and apply your own indexing techniques using llamaindex.
# 
# For more detailed guidance, refer to this [guide](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/multi_modal_videorag_videodb.ipynb).

## 🏃‍♂️ Next Steps
# ---
# 
# In this guide, we built a Simple Multimodal RAG for Videos Using VideoDB, Llamaindex, and Ollama
# 
# You can optimize the pipeline by incorporating more advanced techniques like
# - Optimize Query Transformation
# - More methods to combine retrieved nodes from different modalities
# - Experiment with Different RAG pipelines like Knowledge Graph
# 
# To learn more about Programable Stream feature that we used to create relevant clip checkout [Dynamic Video Stream Guide](https://docs.videodb.io/dynamic-video-stream-guide-44)
# 
# 
# To learn more about Scene Index, explore the following guides:
# 
# - [Quickstart Guide](https://github.com/video-db/videodb-cookbook/blob/main/quickstart/Scene%20Index%20QuickStart.ipynb) 
# - [Scene Extraction Options](https://github.com/video-db/videodb-cookbook/blob/main/guides/scene-index/playground_scene_extraction.ipynb)
# - [Advanced Visual Search](https://github.com/video-db/videodb-cookbook/blob/main/guides/scene-index/advanced_visual_search.ipynb)
# - [Custom Annotation Pipelines](https://github.com/video-db/videodb-cookbook/blob/main/guides/scene-index/custom_annotations.ipynb)

## 👨‍👩‍👧‍👦 Support & Community
# ---
# 
# If you have any questions or feedback. Feel free to reach out to us 🙌🏼
# 
# * [Discord](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdiscord.gg%2Fpy9P639jGz)
# * [GitHub](https://github.com/video-db)
# * [Email](mailto:ashu@videodb.io)

logger.info("\n\n[DONE]", bright=True)